import ast
import collections
import copy
import functools
import hashlib
import inspect
import itertools
import math
import pathlib
import subprocess
import textwrap

import sympy
import triton
from triton.language.extra import libdevice

import ninetoothed.naming as naming
from ninetoothed.cudaifier import Cudaifier
from ninetoothed.language import attribute, call
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor
from ninetoothed.torchifier import Torchifier

CACHE_DIR = pathlib.Path.home() / ".ninetoothed"
CACHE_DIR.mkdir(exist_ok=True)


def _log_debug(message, log_file=None):
    """记录调试日志到文件。
    
    :param message: 要记录的日志消息
    :param log_file: 日志文件路径，如果为 None 则使用默认路径
    """
    if log_file is None:
        log_file = CACHE_DIR / "generation_debug.log"
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        # 如果日志写入失败，静默忽略，避免影响主流程
        pass

class CodeGenerator(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

        device = triton.runtime.driver.active.get_current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)

        self._min_num_elements = 1

        if "max_num_regs" in properties:
            max_innermost_size = 4 * properties["max_num_regs"]
        elif "max_nram_size" in properties:
            max_innermost_size = properties["max_nram_size"]
        else:
            max_innermost_size = 2**18

        self._max_num_elements = max_innermost_size // 8

        # NOTE
        # 获取设备的共享内存限制
        try:
            device = triton.runtime.driver.active.get_current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self._max_shared_mem = properties["max_shared_mem"]
        except Exception:
            # 如果无法获取，使用保守的默认值（48KB，这是大多数 GPU 的最小值）
            self._max_shared_mem = 48 * 1024
        
        # 保留一些共享内存给其他用途（如寄存器溢出等），使用 80% 作为可用限制
        self._available_shared_mem = int(self._max_shared_mem * 0.8)

    def __call__(
        self,
        func,
        caller,
        kernel_name,
        num_warps,
        num_stages,
        max_num_configs,
        prettify,
    ):
        def _get_tree(func):
            func_def = ast.parse(textwrap.dedent(inspect.getsource(func)))

            inliner = _Inliner(func.__globals__)
            inliner.visit(func_def)

            module = ast.Module(body=[func_def], type_ignores=[])

            if inliner.libdevice_used:
                libdevice_alias = ast.alias(
                    name="libdevice", asname=inliner.LIBDEVICE_ALIAS
                )
                libdevice_import = ast.ImportFrom(
                    module="triton.language.extra",
                    names=[libdevice_alias],
                    level=0,
                )

                module.body.insert(0, libdevice_import)

            return module

        def _find_dependencies(func):
            dependencies = set()

            for obj in func.__globals__.values():
                if isinstance(obj, triton.runtime.JITFunction):
                    dependencies.add(obj.src)

            return "\n".join(
                f"@triton.jit\n{dependency}" for dependency in dependencies
            )

        self.launch_func_name = f"launch_{kernel_name}"

        self._caller = caller

        self._num_warps = num_warps

        self._num_stages = num_stages

        self._max_num_configs = max_num_configs

        self._context = inspect.get_annotations(func)

        self._args = list(self._context.values())

        tree = _get_tree(func)

        self.visit(tree)
        Tritonizer().visit(tree)
        _BinOpSimplifier().visit(tree)
        ast.fix_missing_locations(tree)

        if prettify:
            name_collector = _SimplifiedNameCollector()
            name_collector.visit(tree)

        unparsed = ast.unparse(tree).replace("None:", ":").replace(":None", ":")
        dependencies = _find_dependencies(func)
        source = "\n\n".join((unparsed, dependencies)).strip()
        source = source.replace(func.__name__, kernel_name)
        source += "\n"

        if prettify:
            for original, simplified in name_collector.simplified_names.items():
                if simplified not in name_collector.simplified_names:
                    source = source.replace(original, simplified)

            source = subprocess.check_output(
                ["ruff", "format", "-"], input=source, encoding="utf-8"
            )

        cache_file = cache_source(source)

        self.tensors = self._args
        self.kernel_func = self._func_def
        self.launch_func = self._launch

        return str(cache_file)

    def visit_Module(self, node):
        self.generic_visit(node)

        if self._autotune is not None:
            func_with_auto_tuning = f"{Symbol(self._autotune)}({self._func_def.name})"

            node.body.append(
                ast.parse(
                    f"{self._func_name_with_auto_tuning} = {func_with_auto_tuning}"
                )
            )

        node.body.append(self._launch)

        return node

    def visit_FunctionDef(self, node):
        self._func_def = node

        self._func_name_with_auto_tuning = f"{self._func_def.name}_with_auto_tuning"

        self._invariants = {}

        self.generic_visit(node)

        for target, value in reversed(self._invariants.items()):
            node.body.insert(0, ast.Assign(targets=[target.node], value=value.node))

        return node

    def visit_arguments(self, node):
        self.generic_visit(node)

        symbols = {
            name.node.id: name
            for arg in self._args
            for name in arg.names()
            if name != "ninetoothed"
        }
        names = symbols.keys()
        meta_names = {name for name in names if naming.is_meta(name)}
        non_meta_names = {name for name in names if name not in meta_names}
        non_meta_names |= {
            naming.make_next_power_of_2(name)
            for name in non_meta_names
            if naming.is_constexpr(name)
        }

        self._symbols = symbols

        non_meta_names = sorted(non_meta_names)
        meta_names = sorted(meta_names)

        node.args = [
            ast.arg(arg=name)
            if not naming.is_constexpr(name)
            else ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in non_meta_names
        ] + [
            ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in meta_names
        ]

        self._autotune = self._generate_autotune(non_meta_names, meta_names)

        if self._autotune is not None:
            self._func_name = self._func_name_with_auto_tuning
        else:
            self._func_name = self._func_def.name

        self._func_def.decorator_list = [Symbol("triton.jit").node]

        self._launch = self._generate_launch(non_meta_names, meta_names)

        return node

    def visit_Call(self, node):
        def _data_ptr(tensor):
            return Symbol(tensor.source.pointer_string()).node

        def _offsets(tensor, dim=None):
            if dim is None:
                return tensor._last_generated_overall_offsets.node

            offsets = tensor._last_generated_offsets

            if dim < 0:
                dim += tensor.source.ndim

            class _TupleSliceRemover(ast.NodeTransformer):
                def visit_Subscript(self, node):
                    self.generic_visit(node)

                    if isinstance(node.slice, ast.Tuple):
                        return node.value

                    return node

            return _TupleSliceRemover().visit(offsets[dim].node)

        func = node.func
        args = node.args

        if isinstance(func, ast.Attribute):
            if func.attr in ("data_ptr", "offsets"):
                value = func.value

                if self._in_context(value):
                    tensor = self._context[value.id]
                elif isinstance(value, ast.Subscript) and self._in_context(value.value):
                    tensor = self._context[value.value.id]

                self.visit(value)

            if func.attr == "data_ptr":
                return _data_ptr(tensor)

            if func.attr == "offsets":
                # TODO: Add error handling.
                return _offsets(tensor, ast.literal_eval(args[0]) if args else None)

        self.generic_visit(node)

        return node

    def visit_Subscript(self, node):
        if self._in_context(node.value) and isinstance(node.ctx, ast.Load):
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                return self._generate_load(
                    value,
                    indices=node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else (node.slice,),
                )

        self.generic_visit(node)

        return node

    def visit_Attribute(self, node):
        value = node.value

        if isinstance(value, ast.Attribute):
            value = self.visit_Attribute(value)

        if self._in_context(value):
            value = self._context[value.id].dtype

        if isinstance(value, Tensor):
            attr = getattr(value, node.attr)

            if isinstance(attr, Tensor):
                return attr

            if node.attr == "dtype":
                return Symbol(f"{value.source.pointer_string()}.type.element_ty").node

            return Symbol(attr).node

        self.generic_visit(node)

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if self._in_context(node) and isinstance(node.ctx, ast.Load):
            return self._generate_load(self._context[node.id])

        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            if self._in_context(target):
                self.generic_visit(node)

                return ast.Expr(
                    self._generate_store(self._context[target.id], node.value)
                )
            elif (
                isinstance(target, ast.Subscript)
                and self._in_context(target.value)
                and isinstance(target.ctx, ast.Store)
            ):
                value = self._context[target.value.id]

                if isinstance(value, Tensor):
                    return ast.Expr(
                        self._generate_store(
                            value,
                            self.visit(node.value),
                            indices=target.slice.elts
                            if isinstance(target.slice, ast.Tuple)
                            else (target.slice,),
                        )
                    )

        self.generic_visit(node)

        return node

    _NAME_FOR_PID = Symbol("ninetoothed_pid")

    def _in_context(self, node):
        return isinstance(node, ast.Name) and node.id in self._context

    # NOTE
    def _estimate_shared_memory_usage(self, config):
        """估算给定配置下的共享内存使用量（字节）。
        
        :param config: 包含 block_size 配置的字典
        :return: 估算的共享内存使用量（字节）
        """
        _log_debug(f"_estimate_shared_memory_usage: 开始估算，config={config}")
        
        total_memory = 0
        
        # 估算每个 tensor 的内存需求
        # for arg in self._args:
        for arg_idx, arg in enumerate(self._args):
            if arg.ndim == 0:
                _log_debug(f"  arg[{arg_idx}]: ndim=0, 跳过")
                continue
            
            # 获取最内层 tensor 的形状
            innermost = arg.innermost()
            
            # 计算元素数量（用配置替换符号）
            try:
                shape_expr = math.prod(innermost.shape)
                shape_str = str(shape_expr)
                _log_debug(f"  arg[{arg_idx}]: 原始 shape_str={shape_str}")
                
                # 替换配置中的 block_size 符号
                for param, value in config.items():
                    shape_str = shape_str.replace(param, str(value))
                _log_debug(f"  arg[{arg_idx}]: 替换后 shape_str={shape_str}")
                
                # 简化表达式并计算
                num_elements = sympy.simplify(shape_str)
                _log_debug(f"  arg[{arg_idx}]: 简化后 num_elements={num_elements} (type={type(num_elements)})")
                
                # 如果仍然包含符号，使用上界估算
                if isinstance(num_elements, sympy.Symbol) or num_elements.has(sympy.Symbol):
                    # 使用上界进行保守估算
                    for free_symbol in num_elements.free_symbols:
                        symbol_str = str(free_symbol)
                        if symbol_str in self._symbols:
                            symbol = self._symbols[symbol_str]
                            num_elements = num_elements.subs(free_symbol, symbol.upper_bound)
                            _log_debug(f"  arg[{arg_idx}]: 替换符号 {symbol_str} -> {symbol.upper_bound}")
                
                # 转换为数值
                if isinstance(num_elements, (int, float)):
                    num_elements = int(num_elements)
                else:
                    # 如果无法计算，使用上界估算
                    num_elements = int(sympy.N(num_elements))
                
                _log_debug(f"  arg[{arg_idx}]: 最终 num_elements={num_elements}")
                
                # 假设每个元素是 float32（4 字节）
                # 对于其他类型，需要从 tensor 的 dtype 获取，这里简化处理
                element_size = 4  # float32
                tensor_memory = num_elements * element_size
                _log_debug(f"  arg[{arg_idx}]: tensor_memory={tensor_memory} bytes (num_elements={num_elements} * element_size={element_size})")
                
                total_memory += tensor_memory
                
            except Exception as e:
                # 如果计算失败，跳过这个 tensor
                _log_debug(f"  arg[{arg_idx}]: 计算失败，异常={type(e).__name__}: {e}")
                continue
        
        return total_memory

    def _generate_autotune(self, params, meta):
        _log_debug(f"_generate_autotune: 开始生成，params={params}, meta={meta}")
        _log_debug(f"_generate_autotune: _max_shared_mem={self._max_shared_mem} bytes, _available_shared_mem={self._available_shared_mem} bytes")
        
        inequalities = True

        for arg in self._args:
            if arg.ndim == 0:
                continue

            num_elements = sympy.simplify(str(math.prod(arg.innermost().shape)))

            inequalities &= num_elements <= self._max_num_elements
            inequalities &= num_elements >= self._min_num_elements

        values_of_meta_params = []

        for param in meta:
            symbol = self._symbols[param]

            values = range(symbol.lower_bound, symbol.upper_bound + 1)

            if symbol.power_of_two:
                values = tuple(value for value in values if value & (value - 1) == 0)
            else:
                values = tuple(values)

            values_of_meta_params.append(values)
            _log_debug(f"  meta param '{param}': values={values} (lower={symbol.lower_bound}, upper={symbol.upper_bound}, power_of_two={symbol.power_of_two})")

        max_values_of_non_meta_params = {}

        for free_symbol in inequalities.free_symbols:
            symbol_str = str(free_symbol)

            if symbol_str in meta:
                continue

            symbol = self._symbols[symbol_str]

            max_values_of_non_meta_params[symbol_str] = symbol.upper_bound

        block_size_configs = []

        block_size_memory_cache = {}
        total_configs_checked = 0
        total_configs_passed_inequalities = 0
        total_configs_passed_memory = 0
        total_configs_failed_memory = 0

        for values in itertools.product(*values_of_meta_params):
            config = {param: value for param, value in zip(meta, values)}
            total_configs_checked += 1

            if sympy.logic.simplify_logic(
                inequalities.subs(config | max_values_of_non_meta_params)
            ):
                total_configs_passed_inequalities += 1
                # NOTE
                # 添加共享内存检查
                try:
                    shared_mem_usage = self._estimate_shared_memory_usage(config)
                    cache_key = tuple(sorted(config.items()))
                    _log_debug(f"  配置检查: config={config}, shared_mem_usage={shared_mem_usage} bytes, cache_key={cache_key}")
                    
                    if shared_mem_usage <= self._available_shared_mem:
                        block_size_configs.append(config)
                        block_size_memory_cache[cache_key] = shared_mem_usage
                        total_configs_passed_memory += 1
                        _log_debug(f"    -> 通过内存检查，已添加到 block_size_configs (当前总数={len(block_size_configs)})")
                    else:
                        total_configs_failed_memory += 1
                        _log_debug(f"    -> 内存超限，已过滤 (shared_mem_usage={shared_mem_usage} > _available_shared_mem={self._available_shared_mem})")
                except Exception as e:
                    _log_debug(f"    -> 估算异常: {type(e).__name__}: {e}")
                    continue

        _log_debug(f"_generate_autotune: 配置生成统计:")
        _log_debug(f"  总配置数: {total_configs_checked}")
        _log_debug(f"  通过不等式检查: {total_configs_passed_inequalities}")
        _log_debug(f"  通过内存检查: {total_configs_passed_memory}")
        _log_debug(f"  内存超限: {total_configs_failed_memory}")
        _log_debug(f"  最终 block_size_configs 数量: {len(block_size_configs)}")
        _log_debug(f"  缓存大小: {len(block_size_memory_cache)}")

        if not block_size_configs:
            if meta:
                raise ValueError(
                    "Failed to generate auto-tuning. Please check the upper and lower bounds of the symbols."
                )
            else:
                block_size_configs.append({})

        if isinstance(self._num_warps, collections.abc.Iterable):
            num_warps_configs = self._num_warps
        else:
            num_warps_configs = (self._num_warps,)

        if isinstance(self._num_stages, collections.abc.Iterable):
            num_stages_configs = self._num_stages
        else:
            num_stages_configs = (self._num_stages,)

        compiler_configs = tuple(
            {"num_warps": num_warps, "num_stages": num_stages}
            for num_warps, num_stages in itertools.product(
                num_warps_configs, num_stages_configs
            )
        )
        _log_debug(f"_generate_autotune: compiler_configs 数量: {len(compiler_configs)}")

        configs = [
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="ninetoothed", ctx=ast.Load()),
                    attr="Config",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Dict(
                        keys=[
                            ast.Constant(value=param)
                            for param in block_size_config.keys()
                        ],
                        values=[
                            ast.Constant(value=value)
                            for value in block_size_config.values()
                        ],
                    )
                ],
                keywords=[
                    ast.keyword(
                        arg="num_warps",
                        value=ast.Constant(value=compiler_config["num_warps"]),
                    ),
                    ast.keyword(
                        arg="num_stages",
                        value=ast.Constant(value=compiler_config["num_stages"]),
                    ),
                ],
            )
            for block_size_config, compiler_config in itertools.product(
                block_size_configs, compiler_configs
            )
        ]
        _log_debug(f"_generate_autotune: 完整 configs 数量: {len(configs)}")

        # NOTE
        # 在采样之前，按共享内存使用量排序（优先选择内存需求小的配置）
        if self._max_num_configs is not None and len(configs) > self._max_num_configs:
            _log_debug(f"_generate_autotune: 开始采样，max_num_configs={self._max_num_configs}, 当前 configs 数量={len(configs)}")
            
            # 提取配置并估算内存使用量
            config_with_memory = []
            cache_hits = 0
            cache_misses = 0
            
            for config_idx, config_ast in enumerate(configs):
                try:
                    block_size_dict = {}

                    # 从 AST 中提取 block_size 配置
                    if config_ast.args and isinstance(config_ast.args[0], ast.Dict):
                        for key, value in zip(config_ast.args[0].keys, config_ast.args[0].values):
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                                block_size_dict[key.value] = value.value
                    
                    cache_key = tuple(sorted(block_size_dict.items()))
                    memory = block_size_memory_cache.get(cache_key)
                    
                    if memory is None:
                        memory = self._estimate_shared_memory_usage(block_size_dict)
                        _log_debug(f"  采样阶段 config[{config_idx}]: cache_miss, block_size_dict={block_size_dict}, cache_key={cache_key}, memory={memory}")
                        cache_misses += 1
                    else:
                        _log_debug(f"  采样阶段 config[{config_idx}]: cache_hit, block_size_dict={block_size_dict}, cache_key={cache_key}, memory={memory}")
                        cache_hits += 1

                    config_with_memory.append((memory, config_ast))
                except Exception as e:
                    # 如果估算失败，使用一个很大的值
                    _log_debug(f"  采样阶段 config[{config_idx}]: 提取/估算失败，异常={type(e).__name__}: {e}")
                    config_with_memory.append((float('inf'), config_ast))
            
            _log_debug(f"_generate_autotune: 采样阶段统计: cache_hits={cache_hits}, cache_misses={cache_misses}")
            
            # 按内存使用量排序
            config_with_memory.sort(key=lambda x: x[0])
            _log_debug(f"_generate_autotune: 排序完成，前5个配置的内存使用量: {[x[0] for x in config_with_memory[:5]]}")
            
            # 均匀采样，但优先选择内存需求小的
            step = len(config_with_memory) // self._max_num_configs
            _log_debug(f"_generate_autotune: 采样 step={step}")
            
            sampled_indices = []
            configs = []
            for i in range(self._max_num_configs):
                idx = i * step
                sampled_indices.append(idx)
                configs.append(config_with_memory[idx][1])
                _log_debug(f"  采样索引[{i}]: 选择 config_with_memory[{idx}], memory={config_with_memory[idx][0]} bytes")
            
            _log_debug(f"_generate_autotune: 采样完成，最终选择 {len(configs)} 个配置，采样索引={sampled_indices}")

        if len(configs) <= 1:
            _log_debug(f"_generate_autotune: 配置数量 <= 1，返回 None")
            return None
        _log_debug(f"_generate_autotune: 完成，最终返回 {len(configs)} 个配置")
        
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ninetoothed", ctx=ast.Load()),
                attr="autotune",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(
                    arg="configs",
                    value=ast.List(
                        elts=configs,
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="key",
                    value=ast.List(
                        elts=[
                            ast.Constant(value=param)
                            for param in params
                            if not Tensor.pointer_pattern().fullmatch(param)
                        ],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )

    def _generate_launch(self, params, meta):
        non_next_power_of_2_constexpr_params = [
            param
            for param in params
            if naming.is_constexpr(param) and not naming.is_next_power_of_2(param)
        ]
        non_next_power_of_2_constexpr_params_without_prefixes = [
            naming.remove_prefixes(param)
            for param in non_next_power_of_2_constexpr_params
        ]
        next_power_of_2_params = [
            param for param in params if naming.is_next_power_of_2(param)
        ]
        next_power_of_2_params_without_prefixes = [
            naming.remove_prefixes(param) for param in next_power_of_2_params
        ]

        arg_names = [naming.remove_prefixes(arg.source.name) for arg in self._args]

        arg_names += [
            param
            for param in non_next_power_of_2_constexpr_params_without_prefixes
            if not Tensor.size_pattern().fullmatch(param) and param not in arg_names
        ]

        launch = ast.FunctionDef(
            name=self.launch_func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in arg_names],
                kwonlyargs=[],
                defaults=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=ast.Name(id=param_without_prefixes, ctx=ast.Load()),
                )
                for param, param_without_prefixes in zip(
                    non_next_power_of_2_constexpr_params,
                    non_next_power_of_2_constexpr_params_without_prefixes,
                )
            ]
            + [
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=Symbol(
                        f"triton.next_power_of_2({param_without_prefixes})"
                    ).node,
                )
                for param, param_without_prefixes in zip(
                    next_power_of_2_params,
                    next_power_of_2_params_without_prefixes,
                )
            ]
            + [
                ast.Expr(
                    ast.Call(
                        func=ast.Subscript(
                            value=ast.Name(id=self._func_name, ctx=ast.Load()),
                            slice=self._generate_grid(),
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=param, ctx=ast.Load()) for param in params],
                        keywords=[
                            ast.keyword(
                                arg="num_warps",
                                value=ast.Constant(value=self._num_warps),
                            ),
                            ast.keyword(
                                arg="num_stages",
                                value=ast.Constant(value=self._num_stages),
                            ),
                        ]
                        if self._autotune is None
                        else [],
                    )
                )
            ],
            decorator_list=[],
        )

        class MetaEncloser(ast.NodeTransformer):
            def __init__(self, meta):
                self._meta = meta

            def visit_Name(self, node):
                self.generic_visit(node)

                if node.id in self._meta:
                    return ast.Subscript(
                        value=ast.Name(id="meta", ctx=ast.Load()),
                        slice=ast.Constant(value=node.id),
                        ctx=ast.Load(),
                    )

                return node

        MetaEncloser(meta).visit(launch)

        if self._caller == "torch":
            Torchifier().visit(launch)
        elif self._caller == "cuda":
            Cudaifier().visit(launch)
        else:
            raise ValueError(f"Unsupported caller: `{self._caller}`.")

        return launch

    def _generate_grid(self):
        num_elements = functools.reduce(lambda x, y: x * y, self._args[0].shape)

        grid = ast.parse(f"lambda meta: ({num_elements},)", mode="eval").body

        self.raw_grid = copy.deepcopy(grid)

        return grid

    def _generate_load(self, tensor, indices=()):
        if tensor.ndim == 0:
            return Symbol(tensor.source.name).node

        pointers, mask = self._generate_pointers_and_mask(tensor, indices)
        other = type(self)._generate_other(tensor)

        return call("load", pointers, mask=mask, other=other).node

    def _generate_store(self, tensor, value, indices=()):
        pointers, mask = self._generate_pointers_and_mask(tensor, indices)

        return call("store", pointers, value, mask=mask).node

    def _generate_pointers_and_mask(self, tensor, indices):
        indices = [Symbol(index) for index in self._complete_indices(tensor, indices)]

        name_for_pointers = type(self)._name_for_pointers(tensor)
        self._invariants[name_for_pointers] = Symbol(tensor.source.pointer_string())

        overall_offsets, mask = type(self)._generate_overall_offsets_and_mask(
            tensor, indices
        )

        pointers = name_for_pointers + overall_offsets

        return pointers, mask

    def _complete_indices(self, tensor, indices):
        return (
            tuple(self._generate_pid_indices(tensor))
            + tuple(indices)
            + tuple(type(self)._generate_innermost_indices(tensor))
        )

    def _generate_pid_indices(self, tensor):
        self._invariants[type(self)._NAME_FOR_PID] = call("program_id", 0)

        indices = list(Tensor._unravel_index(type(self)._NAME_FOR_PID, tensor.shape))

        for dim, index in enumerate(indices):
            name = type(self)._name_for_index(tensor, dim)
            self._invariants[name] = index
            indices[dim] = name

        if tensor.source.jagged_dim is not None:
            seq_len_name = Symbol(tensor.source.seq_len_string())
            max_seq_len_name = Symbol(tensor.source.max_seq_len_string())

            for size in tensor.shape:
                size.find_and_replace(seq_len_name, max_seq_len_name)

            offsets_name = Symbol(tensor.source.offsets_string())
            batch_dim_index_name = type(self)._name_for_index(tensor, 0)
            seq_start_name = type(self)._name_for_seq_start(tensor)
            seq_end_name = type(self)._name_for_seq_end(tensor)

            self._invariants[seq_start_name] = call(
                "load", offsets_name + batch_dim_index_name
            )
            self._invariants[seq_end_name] = call(
                "load", offsets_name + batch_dim_index_name + 1
            )
            self._invariants[seq_len_name] = seq_end_name - seq_start_name

        return tuple(indices)

    @staticmethod
    def _generate_other(tensor):
        other = tensor.source.other

        if isinstance(other, float) and not math.isfinite(other):
            return f"float('{other}')"

        return other

    @staticmethod
    def _generate_slices(tensor, dim):
        return tuple(
            slice(None) if target_dim == dim else None
            for target_dim in tensor.innermost().target_dims
        )

    @staticmethod
    def _generate_overall_offsets_and_mask(tensor, indices):
        indices = list(indices)

        offsets, mask = CodeGenerator._generate_offsets_and_mask(tensor, indices)

        tensor._last_generated_offsets = offsets

        overall_offsets = sum(
            offsets[source_dim] * Symbol(tensor.source.stride_string(source_dim))
            for source_dim in range(tensor.source.ndim)
        )

        if tensor.source.jagged_dim is not None:
            overall_offsets += CodeGenerator._name_for_seq_start(tensor) * Symbol(
                tensor.source.stride_string(tensor.source.jagged_dim)
            )

        tensor._last_generated_overall_offsets = overall_offsets

        return overall_offsets, mask

    @staticmethod
    def _generate_offsets_and_mask(tensor, indices):
        offsets = [Symbol(0) for _ in range(tensor.source.ndim)]

        tensor.source._mask = Symbol(True)

        curr = tensor
        start = 0

        while isinstance(curr, type(tensor)):
            stop = start + curr.ndim
            curr_indices = indices[start:stop]

            curr._inputs = [curr_indices]

            start = stop
            curr = curr.dtype

        for level in reversed(tensor._levels):
            for tensor_ in level:
                tensor_.offsets()

        for dim, offset in enumerate(tensor.source._outputs[0]):
            offsets[dim] += offset

        curr = tensor

        while isinstance(curr, type(tensor)):
            curr._inputs.clear()

            curr = curr.dtype

        return offsets, tensor.source._mask

    @staticmethod
    def _generate_innermost_indices(tensor, use_power_of_2_sizes=True):
        class _NextPowerOfTwoMaker(ast.NodeTransformer):
            def visit_Name(self, node):
                name = node.id

                if not naming.is_meta(name):
                    next_power_of_2_name = naming.make_next_power_of_2(name)

                    return ast.Name(id=next_power_of_2_name, ctx=ast.Load())

                return self.generic_visit(node)

        indices = []

        for size, target_dim in zip(
            tensor.innermost().shape, tensor.innermost().target_dims
        ):
            if use_power_of_2_sizes:
                size = _NextPowerOfTwoMaker().visit(Symbol(copy.deepcopy(size)).node)

            indices.append(
                call("arange", 0, size)[
                    CodeGenerator._generate_slices(tensor, target_dim)
                ]
            )

        return tuple(indices)

    @staticmethod
    def _name_for_pointers(tensor):
        return Symbol(f"{tensor.source.name}_pointers")

    @staticmethod
    def _name_for_offsets(tensor, source_dim, target_dim):
        return Symbol(f"{tensor.source.name}_offsets_{source_dim}_{target_dim}")

    @staticmethod
    def _name_for_seq_start(tensor):
        return Symbol(f"{tensor.source.name}_seq_start")

    @staticmethod
    def _name_for_seq_end(tensor):
        return Symbol(f"{tensor.source.name}_seq_end")

    @staticmethod
    def _name_for_index(tensor, dim):
        return Symbol(f"{tensor.source.name}_index_{dim}")


class Tritonizer(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)

        node.body.insert(0, ast.Import(names=[ast.alias(name="triton.language")]))
        node.body.insert(0, ast.Import(names=[ast.alias(name="triton")]))

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id == "ninetoothed" or "ninetoothed." in node.id:
            node.id = node.id.replace("ninetoothed", "triton")

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "jit"
        ):
            return ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load()
            )

        return node


def cache_source(source):
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{digest}.py"

    if not cache_file.exists():
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(source)

    return cache_file


class _Inliner(ast.NodeTransformer):
    LIBDEVICE_ALIAS = naming.auto_generate("libdevice")

    def __init__(self, globals):
        self.libdevice_used = False

        self._globals = globals

        self._count = 0

    def visit(self, node):
        def _find_aliases():
            aliases = {}

            for name, value in self._globals.items():
                if inspect.ismodule(value):
                    if value is libdevice:
                        aliases[name] = self.LIBDEVICE_ALIAS
                        self.libdevice_used = True

                        continue

                    aliases[name] = value.__name__

            return aliases

        node = super().visit(node)

        alias_restorer = _AliasRestorer(_find_aliases())

        if isinstance(node, list):
            node = [alias_restorer.visit(item) for item in node]
        else:
            node = alias_restorer.visit(node)

        return node

    def visit_Expr(self, node):
        value, stmts = self._inline_expr(node.value)
        node.value = value
        node = self.generic_visit(node)

        if stmts:
            if isinstance(value, ast.Constant) and value.value is None:
                return stmts

            return stmts + [node]

        return node

    def visit_Assign(self, node):
        value, stmts = self._inline_expr(node.value)
        node.value = value
        node = self.generic_visit(node)

        if stmts:
            return stmts + [node]

        return node

    def visit_Return(self, node):
        if node.value:
            value, stmts = self._inline_expr(node.value)
            node.value = value

            if stmts:
                return stmts + [node]

        return node

    def _inline_expr(self, expr):
        def _inline_list(lst):
            new_list = []
            new_stmts = []

            for expr in lst:
                expr, stmts = self._inline_expr(expr)

                new_list.append(expr)
                new_stmts.extend(stmts)

            return new_list, new_stmts

        def _inline_field(field):
            if isinstance(field, ast.AST):
                return self._inline_expr(field)

            return field, []

        if isinstance(expr, ast.Call):
            new_expr, new_stmts = self._inline_call(expr)

            if new_expr is not None:
                return new_expr, new_stmts

        new_stmts = []

        for field, value in ast.iter_fields(expr):
            if isinstance(value, list):
                new_value, new_stmts = _inline_list(value)
            else:
                new_value, new_stmts = _inline_field(value)

            setattr(expr, field, new_value)
            new_stmts.extend(new_stmts)

        return expr, new_stmts

    def _inline_call(self, node):
        class _ParameterReplacer(ast.NodeTransformer):
            def __init__(self, mapping):
                self._mapping = mapping

            def visit_Name(self, node):
                return self._mapping.get(node.id, node)

        class _LocalVariableRenamer(ast.NodeTransformer):
            def __init__(self, prefix, local_vars):
                self._prefix = prefix

                self._local_vars = local_vars

            def visit_Name(self, node):
                if node.id in self._local_vars:
                    node.id = f"{self._prefix}{node.id}"

                return node

            def visit_arg(self, node):
                return node

        def _resolve_function(node, globals):
            if isinstance(node, ast.Name):
                return globals.get(node.id)

            if isinstance(node, ast.Attribute):
                obj = _resolve_function(node.value, globals)

                if obj is not None:
                    return getattr(obj, node.attr, None)

            return None

        def _get_source(func):
            try:
                return inspect.getsource(func)
            except TypeError:
                return None

        def _find_function_definition(source):
            finder = _FunctionDefFinder(func.__name__)
            finder.visit(ast.parse(source))

            return finder.result

        def _find_assigned_names(stmts):
            class _AssignedNameFinder(ast.NodeVisitor):
                def __init__(self):
                    self.result = set()

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        self.result.add(node.id)

            names = set()

            for stmt in stmts:
                finder = _AssignedNameFinder()
                finder.visit(stmt)
                names |= finder.result

            return names

        def _make_temporary():
            prefix = f"{naming.auto_generate(f'temporary_{self._count}')}_"
            self._count += 1

            return prefix

        func = _resolve_function(node.func, self._globals)

        if func is None:
            return None, []

        source = _get_source(func)

        if source is None:
            return None, []

        func_def = _find_function_definition(source)

        if func_def is None:
            return None, []

        if inspect.getmodule(func) is libdevice:
            return None, []

        param_names = [arg.arg for arg in func_def.args.args]

        mapping = {param: arg for param, arg in zip(param_names, node.args)}
        param_replacer = _ParameterReplacer(mapping)
        body = [param_replacer.visit(stmt) for stmt in func_def.body]

        local_vars = _find_assigned_names(body) - set(param_names)
        prefix = _make_temporary()
        local_var_renamer = _LocalVariableRenamer(prefix, local_vars)
        body = [local_var_renamer.visit(stmt) for stmt in body]

        inlined_body = []

        inliner = _Inliner(func.__globals__)

        for stmt in body:
            inlined_stmt = inliner.visit(stmt)

            if isinstance(inlined_stmt, list):
                inlined_body.extend(inlined_stmt)
            else:
                inlined_body.append(inlined_stmt)

        if not inlined_body or not isinstance(inlined_body[-1], ast.Return):
            return ast.Constant(value=None), inlined_body

        ret = inlined_body.pop()
        temp = _make_temporary()
        assignment = ast.Assign(
            targets=[ast.Name(id=temp, ctx=ast.Store())], value=ret.value
        )
        inlined_body.append(assignment)

        return ast.Name(id=temp, ctx=ast.Load()), inlined_body


class _BinOpSimplifier(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.Mult):
            left = Symbol(node.left)
            right = Symbol(node.right)

            if left == 1:
                return node.right

            if right == 1:
                return node.left

        return node


class _SimplifiedNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.simplified_names = {}

    def visit_Name(self, node):
        self.generic_visit(node)

        self.simplified_names[node.id] = naming.remove_prefixes(node.id)


class _AliasRestorer(ast.NodeTransformer):
    def __init__(self, aliases):
        super().__init__()

        self._aliases = aliases

        self._redefined = set()

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._redefined.add(target.id)

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        original_redefined = self._redefined.copy()

        self.generic_visit(node)

        self._redefined = original_redefined

        return node

    def visit_Name(self, node):
        if node.id in self._redefined:
            return node

        if node.id in self._aliases:
            return ast.Name(id=self._aliases[node.id], ctx=node.ctx)

        return node


class _FunctionDefFinder(ast.NodeVisitor):
    def __init__(self, name):
        self._name = name

        self.result = None

    def visit_FunctionDef(self, node):
        if node.name == self._name:
            self.result = node

        self.generic_visit(node)

import ast
import types
import inspect
from typing import Callable


class ASTVisitor(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        # Hierarchy of read/written variable names
        self.locals_r = []
        self.locals_w = []
        self.disarm = False
        self.loop_ctr = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.disarm:
            # Process only the outermost function
            self.disarm = True

            # Add function parameters to self.locals_w
            locals_w = set()
            for o1 in (node.args.args, node.args.posonlyargs, node.args.kwonlyargs):
                for o2 in o1:
                    locals_w.add(o2.arg)
            self.locals_r.append(set())
            self.locals_w.append(locals_w)

            node = self.generic_visit(node)

        return node

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.locals_r[-1].add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.locals_w[-1].add(node.id)
        return node

    def visit_While(self, node: ast.While):
        self.locals_r.append(set())
        self.locals_w.append(set())

        # 1. Names of generated functions
        loop_name = "_loop"
        cond_name = loop_name + "_cond"
        step_name = loop_name + "_step"
        state_name = loop_name + "_state"
        local_name = "_s"

        # 2. Process the loop condition separately
        node.body, body = [], node.body
        node_test = self.generic_visit(node).test
        locals_r_cond = set(self.locals_r[-1])
        assert len(self.locals_w[-1]) == 0

        # 3. Process the loop body separately
        node.body = body
        node.test = ast.Constant(value="True")
        node_body = self.generic_visit(node).body

        locals_r, locals_w = self.locals_r.pop(), self.locals_w.pop()
        parent_r, parent_w = set(), set()

        for s in self.locals_r:
            parent_r |= s
        for s in self.locals_w:
            parent_w |= s

        # 4. Compute data mapping:
        # 4a. Don't import globals into state data structure
        temp = set()
        for k in locals_r | locals_r_cond:
            if k not in locals_w and k not in parent_w:
                temp.add(k)
        locals_r -= temp
        locals_r_cond -= temp

        # 4b. Don't store loop temporaries. We consider temporaries
        # to be variables that haven't been defined before the loop
        # (which would be undefined if the loop condition is False)
        temp = set()
        for k in locals_w:
            if not k in parent_w:
                temp.add(k)
        locals_r -= temp
        locals_w -= temp

        locals_rw = locals_r | locals_w
        locals_rw_both = locals_rw | locals_r_cond

        locals_rw_both = sorted(locals_rw_both)
        locals_rw = sorted(locals_rw)
        locals_r = sorted(locals_r)
        locals_w = sorted(locals_w)

        # 5. Generate a function representing the loop condition
        #    .. which takes a state dictionary as input
        func_args = ast.arguments(
            args=[ast.arg(local_name)],
            posonlyargs=[],
            kwonlyargs=[],
            defaults=[],
            kw_defaults=[],
        )

        # 5a. Generate statement to import loop state into local scope
        load, store, delete = ast.Load(), ast.Store(), ast.Del()
        state = ast.Name(local_name, ctx=load)
        cond_import_state = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=k, ctx=store) for k in locals_r_cond], ctx=store
                )
            ],
            value=ast.Tuple(
                elts=[
                    ast.Subscript(value=state, slice=ast.Constant(value=k), ctx=load)
                    for k in locals_r_cond
                ],
                ctx=load,
            ),
        )

        # 5b. Generate the actual function
        cond_func = ast.FunctionDef(
            name=cond_name,
            args=func_args,
            body=[
                cond_import_state,
                ast.Return(value=node_test),
            ],
            decorator_list=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        # 6. Generate a function representing the loop body
        # 6a. Generate statement to import loop state into local scope
        step_import_state = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=k, ctx=store) for k in locals_rw], ctx=store
                )
            ],
            value=ast.Tuple(
                elts=[
                    ast.Subscript(value=state, slice=ast.Constant(value=k), ctx=load)
                    for k in locals_rw
                ],
                ctx=load,
            ),
        )

        # 6b. Generate statement to export loop state from local scope
        step_export_state = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        ast.Subscript(
                            value=state, slice=ast.Constant(value=k), ctx=store
                        )
                        for k in locals_w
                    ],
                    ctx=store,
                )
            ],
            value=ast.Tuple(
                elts=[ast.Name(id=k, ctx=load) for k in locals_w], ctx=load
            ),
        )

        # 6c. Generate the actual function
        step_func = ast.FunctionDef(
            name=step_name,
            args=func_args,
            body=[step_import_state, *node_body, step_export_state],
            decorator_list=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        # 7. Import the Dr.Jit while loop
        import_stmt = ast.ImportFrom(
            module="drjit",
            names=[ast.alias(name="while_loop", asname=loop_name)],
            level=0,
        )

        # 8. Generate statement to create the loop state object
        loop_export_state = ast.Assign(
            targets=[ast.Name(id=state_name, ctx=store)],
            value=ast.Dict(
                keys=[ast.Constant(value=k) for k in locals_rw_both],
                values=[ast.Name(id=k, ctx=load) for k in locals_rw_both],
            ),
        )

        # 9. Call the while loop function
        while_expr = ast.Expr(
            value=ast.Call(
                func=ast.Name(id=loop_name, ctx=load),
                args=[
                    ast.Name(id=state_name, ctx=load),
                    ast.Name(id=cond_name, ctx=load),
                    ast.Name(id=step_name, ctx=load),
                ],
                keywords=[],
                lineno=node.lineno,
                col_offset=node.col_offset,
            ),
        )

        # 10. Export loop state back into the local frame
        state_var = ast.Name(id=state_name, ctx=load)
        loop_import_state = ast.Assign(
            targets=[
                ast.Tuple(elts=[ast.Name(id=k, ctx=store) for k in locals_w], ctx=store)
            ],
            value=ast.Tuple(
                elts=[
                    ast.Subscript(
                        value=state_var, slice=ast.Constant(value=k), ctx=load
                    )
                    for k in locals_w
                ],
                ctx=load,
            ),
        )

        # 11. Some comments (as strings) to delineate processed parts of the AST
        comment_start = ast.Expr(
            ast.Constant("------- loop transformed by dr.function -------")
        )
        comment_mid = ast.Expr(
            ast.Constant("----------- invoke dr.while_loop --------------")
        )
        comment_end = ast.Expr(
            ast.Constant("-----------------------------------------------")
        )

        # 12. Delete local variables created while processing the loop
        cleanup = ast.Delete(
            targets=[
                ast.Name(id=loop_name, ctx=delete),
                ast.Name(id=cond_name, ctx=delete),
                ast.Name(id=step_name, ctx=delete),
                ast.Name(id=state_name, ctx=delete),
            ]
        )

        self.locals_r[-1].update(locals_r)
        self.locals_w[-1].update(locals_w)

        return (
            comment_start,
            cond_func,
            step_func,
            comment_mid,
            import_stmt,
            loop_export_state,
            while_expr,
            loop_import_state,
            cleanup,
            comment_end,
        )


def function(f: Callable = None, print_ast: bool = False, print_code: bool = False):
    """
    Decorator for vectorized loops and conditionals.

    This decorator provides *syntax sugar*. It allows users to write natural
    Python code that it then turns into native Dr.Jit constructs. It *does not
    compile* or otherwise change the behavior of the function.

    The :py:func:`@drjit.function <drjit.function>` decorator introduces two
    specific changes:

    1. It rewrites ``while`` loops so that they still work when the loop
       condition is a Dr.Jit array. In that case, each element of the array
       may want to run a different number of loop iterations.

    2. Analogously, it rewrites ``if`` statements so that they work when the
    condition expression is a Dr.Jit array. In that case, only a subset of
    array elements may want to execute the body of the ``if`` statement.

    Other control flow statements are unaffected. The transformed function may
    call other functions, whether annotated by :py:func:`drjit.function` or
    not. The introduced transformations only affect the annotated function.

    Internally, function turns ``while`` loops and ``if`` statements into calls
    to :py:func:`drjit.while_loop` and :py:func:`drjit.if_stmt`. It is tedious
    to write large programs in this way, which is why the decorator exists.

    For example, consider the following function that raises a floating point
    array to an integer power. The resulting code looks natural thanks to the
    :py:func:`@drjit.function <drjit.function>` decorator.

    .. code-block:: python

       import drjit as dr
       from drjit.cuda import Int, Float

       @dr.function
       def ipow(x: Float, n: Int):
           result = Float(1)

           while n != 0:
               if n & 1 != 0:
                   result *= x
               x *= x
               n >>= 1

           return result

    This (roughly) expands into the following native code that determines
    relevant state variables and wraps conditionals and blocks into functions.
    These transformations enable symbolic compilation and automatic derivative
    propagation in forward and reverse mode.

    .. code-block:: python

       def ipow(x: Float, n: Int):
           # Loop condition wrapped into a callable for ``drjit.while_loop``
           def loop_cond(n, x, result):
               return n != 0

           # Loop body wrapped into a callable for ``drjit.while_loop``
           def loop_body(n, x, result):
               # Conditional expression wrapped into callable for drjit.if_stmt
               def if_cond(n, x, result):
                   return n & 1 != 0

               # Conditional body wrapped into callable for drjit.if_stmt
               def if_body(n, x, result):
                   result *= x

                   # Return updated state following conditional stmt
                   return (n, x, result)

               # Map the 'n', 'x', and 'result' variables though the conditional
               n, x, result = dr.if_stmt(
                   (n, x, result),
                   if_cond,
                   if_body,
               )

               # Rest of the loop body copy-pasted (no transformations needed here)
               x *= x
               n >>= 1

               # Return updated loop state
               return (n, x, result)

           result = Float(1)

           # Execute the loop and assign the final loop state to local variables
           n, x, result = dr.while_loop(
               (n, x, result)
               loop_cond,
               loop_body
           )

           return result

    The :py:func:`@drjit.function <drjit.function>` decorator preserves line
    number information so that debugging works and exeptions/error messages are
    tied to the right locations in the corresponding *untransformed* function.

    There are two additional keyword arguments `print_ast` and `print_code`
    that are both disabled by default. Set them to ``True`` to inspect the
    function before/after the transformation, either using an AST dump or via
    generated Python code.

    .. code-block:: python

       @dr.function(print_code=True)
       def ipow(x: Float, n: Int):
           # ...

    Note that the functions :py:func:`if_stmt` and :py:func:`while_loop` even
    work when the loop condition is *scalar* (a Python `bool`). Since they
    don't do anything special in that case, you may want to avoid the
    transformation altogether. You can provide such control flow hints using
    :py:func:`drjit.hint`. Other hints can also be provided to request
    compilation using evaluated/symbolic mode, or to specify a maximum number
    of loop iteration for reverse-mode automatic differentiation.

    .. code-block:: python

       @dr.function
       def foo():
           i = 10 # 'i' is a Python 'int' and therefore does not need special
                  # handling introduced by @dr.function

           # Disable the transformation by @dr.function to avoid overheads
           while dr.hint(i < 10, scalar=True):
               i += 1
    """

    if f is None:

        def wrapper(f2):
            return function(f2, print_ast, print_code)

        return wrapper

    source = inspect.getsource(f)
    old_ast = ast.parse(source)
    new_ast = old_ast
    if print_ast:
        print(f"Input AST\n---------\n{ast.dump(old_ast, indent=4)}\n")
    if print_code:
        print(f"Input code\n----------\n{ast.unparse(old_ast)}\n")

    new_ast = ASTVisitor().visit(old_ast)
    new_ast = ast.fix_missing_locations(new_ast)

    if print_ast:
        print(f"Output AST\n----------\n{ast.dump(new_ast, indent=4)}\n")
    if print_code:
        print(f"Output code\n-----------\n{ast.unparse(new_ast)}\n")

    old_code = f.__code__
    ast.increment_lineno(new_ast, old_code.co_firstlineno - 1)
    try:
        new_code = compile(new_ast, old_code.co_filename, "exec")
    except BaseException as e:
        raise RuntimeError(
            "The following transformed AST generated by "
            "@drjit.function could not be compiled:\n\n%s" % ast.unparse(new_ast)
        ) from e
    new_code = next(
        (x for x in new_code.co_consts if isinstance(x, types.CodeType)), None
    )
    return types.FunctionType(new_code, f.__globals__)


def hint(
    arg: object,
    /,
    *,
    scalar=None,
    evaluate=None,
    symbolic=None,
    max_iterations=None,
    name=None,
) -> object:
    """
    Within ordinary Python code, this function is unremarkable: it returns the
    positional-only argument `arg` while ignoring any specified keyword
    arguments.

    The main purpose of :py:func:`drjit.hint()` is to provide *hints* that
    influence the transformation performed by the :py:func:`@drjit.function
    <drjit.function>` decorator. The following kinds of hints are supported:

    1. Disabling code transformations for scalar code.

       Wrap the condition of a ``while`` loop or ``if`` statement in a hint
       that specifies ``scalar=True`` to entirely disable the transformation:

       .. code-block:: python

          i = 0
          while dr.hint(i < 10, scalar=True):
             # ...

    2. Force *evaluated* mode.

       Wrap the condition of a ``while`` loop or ``if`` statement in a hint
       that specifies ``evaluate=True`` to force execution in *evaluated* mode.
       For a loop, this is, e.g., analogous to wrapping the code in

       .. code-block:: python

          while dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
             # ...

       Refer to the discussion of :py:func:`drjit.while_loop`,
       :py:attr:`drjit.JitFlag.SymbolicLoops` :py:func:`drjit.if_stmt`, and
       :py:attr:`drjit.JitFlag.SymbolicConditionals` for details.

    3. Force *symbolic* mode.

       Wrap the condition of a ``while`` loop or ``if`` statement in a hint
       that specifies ``symbolic=True`` to force execution in *symbolic* mode.
       For a loop, this is, e.g., analogous to wrapping the code in

       .. code-block:: python

          while dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, True):
             # ...

       Refer to the discussion of :py:func:`drjit.while_loop`,
       :py:attr:`drjit.JitFlag.SymbolicLoops` :py:func:`drjit.if_stmt`, and
       :py:attr:`drjit.JitFlag.SymbolicConditionals` for details.

    4. Specify a maximum number of loop iterations for reverse-mode
       automatic differentiation.

       Naive reverse-mode differentiation of loops (unless replaced by a
       smarter problem-specific strategy via :py:class:`drjit.custom` and
       :py:class:`drjit.CustomOp`) requires allocation of a large buffer that
       holds loop state for all iterations.

       Dr.Jit requires an upper bound on the maximum number of loop iterations
       so that it can allocate such a buffer, which can be provided via this
       hint. Otherwise, reverse-mode differentiation of loops will fail with an
       error message.

    5. Provide a descriptive name.

       Specify the `name` parameter (of type ``str``). Dr.Jit will include this
       name as a comment in the generated intermediate representation, which
       can be helpful when debugging the compilation of large programs.

    """
    return arg

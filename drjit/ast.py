import ast
import types
import inspect


class ASTVisitor(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.locals_r = set()
        self.locals_w = set()
        self.disarm = False
        self.loop_ctr = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not self.disarm:
            for o1 in (node.args.args, node.args.posonlyargs, node.args.kwonlyargs):
                for o2 in o1:
                    self.locals_w.add(o2.arg)
            # Process only the outermost function
            self.disarm = True
            node = self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.locals_r.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.locals_w.add(node.id)
        return node

    def visit_While(self, node: ast.While):
        locals_r, self.locals_r = self.locals_r, set()
        locals_w, self.locals_w = self.locals_w, set()

        # 1. Names of generated functions
        loop_name = "_loop"
        cond_name = loop_name + "_cond"
        step_name = loop_name + "_step"
        state_name = loop_name + "_state"
        local_name = "_s"

        # 2. Process the loop condition separately
        node.body, body = [], node.body
        node_test = self.generic_visit(node).test
        locals_r_cond = set(self.locals_r)
        assert len(self.locals_w) == 0
        self.locals_r.clear()

        # 3. Process the loop body separately
        node.body = body
        node.test = ast.Constant(value="True")
        node_body = self.generic_visit(node).body

        self.locals_r, locals_r = locals_r, self.locals_r
        self.locals_w, locals_w = locals_w, self.locals_w

        # 4. Compute data mapping:
        # 4a. Don't import globals into state data structure
        temp = set()
        for k in locals_r | locals_r_cond:
            if k not in locals_w and k not in self.locals_w:
                temp.add(k)
        locals_r -= temp
        locals_r_cond -= temp

        # 4b. Don't store loop temporaries. We consider temporaries
        # to be variables that haven't been defined before the loop
        # (which would be undefined if the loop condition is False)
        temp = set()
        for k in locals_w:
            if not k in self.locals_w:
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
            col_offset=node.col_offset
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
            col_offset=node.col_offset
        )

        # 7. Import the Dr.Jit while loop
        import_stmt = ast.ImportFrom(
            module="drjit",
            names=[ast.alias(name="while_loop", asname=loop_name)],
            level=0,
        )

        # 8. Generate statement to create the loop state object
        loop_export_state = ast.Assign(
            targets=[
                ast.Name(id=state_name, ctx=store)
            ],
            value=ast.Dict(
                keys =[
                    ast.Constant(value=k) for k in locals_rw_both
                ],
                values=[
                    ast.Name(id=k, ctx=load) for k in locals_rw_both
                ]
            ),
        )

        # 9. Call the while loop function
        while_expr = ast.Assign(
            targets=[ast.Name(id=state_name, ctx=store)],
            value=ast.Call(
                func=ast.Name(id=loop_name, ctx=load),
                args=[
                    ast.Name(id=state_name, ctx=load),
                    ast.Name(id=cond_name, ctx=load),
                    ast.Name(id=step_name, ctx=load),
                ],
                keywords=[],
                lineno=node.lineno,
                col_offset=node.col_offset
            ),
        )

        # 10. Export loop state back into the local frame
        state_var = ast.Name(id=state_name, ctx=load)
        loop_import_state = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=k, ctx=store) for k in locals_w], ctx=store
                )
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

        self.locals_r.update(locals_r)
        self.locals_w.update(locals_w)

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


def function(f=None, print_ast=False, print_code=False):
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

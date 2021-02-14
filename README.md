# BVP-Solver
Solve any Boundary Value Problem with any type of Boundary Condition with this general Boundary Value Problem Interface implemented in Python!

## Implementation ⌨️
This interface combines the excellent capabilities of both Sympy and Scipy within an unique easy-to-use class. As explained in the next section, the user may choose one option to introduce the Ordinary Differential Equation (ODE):
  1. Introduce the full expression of the ODE as a string.
  2. Introduce the system as an array of first order ODE equations.
 
In the first case, where the user introduces the full ODE as a string, the latter is parsed using Sympy's function `parse_expr` function, where y is interpreted as the function to be solved (y = y(x)) and x is the independent variable. Once the expression is correctly parsed, the order of the ODE is determined using the static method `get_ode_order` from this class, which is just a wrapper of Sympy's function `ode_order`. The previous step is required because now it is necessary to express the ODE as a system of First Order ODEs using the built-in method `create_system`. This is achieved with Sympy's dummy variables. Once the system has been created, a lambda expression is created for each of the equations conforming the system, so they can be read by the Scipy's solver.

For the second case where the user introduces the system, this is simply loaded to the solver and internally translated using the appropriate dummy variables, so it can be readily translated to a lambda expression which can be interpreted by the Scipy's solver.

## Usage 👨‍💻👩‍💻
In just a simple steps, you will be able to solve even the most complex BVP once the class has been initialized:

  1. Introduce either a string containing the full ODE or an array with the system. The notation and built-in functions introduced within the string should match             the notation followed by Sympy. For example, to express a derivative of first order with respect to x, one may use the following syntax: `ode = "y.diff(x, 1)"`, where 1 indicates the ODE order (this can be changed accordingly). One may also introduce functions like `exp`, `sin`, `cos`, etc. To see a list of all built-in functions, check the following [Sympy's documentation](https://docs.sympy.org/latest/modules/functions/index.html#contents). In case you want to load the full expression of the ODE, use the method `define_ode`. Otherwise, use `define_system`.
  2. Create the initial mesh containing the points where the ODE will be solved. The final number of nodes used may vary with respect to the original mesh to achieve the desired tolerance/accuracy. This mesh can be created using Numpy's `linspace` function, where the initial and final points **must** be the boundary points of the domain. Once the array containing the mesh points is created, this can be loaded to the solver by using the method `load_mesh`.
  3. Create and load an initial guess (optional). This can be accomplished by creating a zeros array using Numpy's functionalities, where the dimensions must be (`ode_order`, `len(mesh)`), where `len(mesh)`is the number of points used to initialize the mesh from previous step. If desired, this initial guess may be loaded using the method `load_initial_guess`.
  4. Create and load the boundary conditions. In order to do so, a dictionary is required whose keys **must** be `"a"`and `"b"`, where 'a' represents the boundary condition to be imposed at the first point of the domain (initial point) and 'b' is the final point of the domain. The values of this first dictionary must be also a dictionary whose key must be a string containing the expression of the boundary condition, using the same notation as the one explained in Step 1. For example, say we need to impose a boundary condition at the initial point of the domain of `y(a) = 0` and also the following boundary condition for the final point of the domain `y'(b) = 0`, where `'`indicates a first order derivative with respect to x. These boundary conditions may be created using the following implementation: `bcs = {"a": {"y": 0}, "b": {"y.diff(x, 1)": 0}}`. There are infinitely combinations of boundary conditions where the user may also introduce previously computed constants. In that case, one may format the string using `f"{k}*y.diff(x, 1)"`, where `k` is a constant. This can also be applied when defining the ODE in Step 1. Finally, once the boundary conditions have been properly initialized, these should be loaded using the method `load_bcs`.
  5. You are ready to go! Simply call the `solve` method and Scipy's object containing the solution will be returned. For more information regarding the core solver of this interface, check [Scipy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html). Notice that one may introduce any kwarg in the `solve` method as long as it is accepted by the Scipy's solver.

## Examples.
Many useful examples can be found within the interface class file.

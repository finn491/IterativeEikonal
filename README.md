# EikIVP
Solve Eikonal PDEs on $`\mathbb{R}^2`$, $`\mathbb{M}_2 \coloneqq \mathbb{R}^2 \times S^1`$, and $`\mathbb{W}_2 \coloneqq S^2 \times S^1`$ using the iterative method described in Bekkers et al. (2015) "A PDE Approach to Data-Driven Sub-Riemannian Geodesics in $`\mathrm{SE}(2)`$" [[1]](#1). 

The viscosity solutions of the Eikonal equation is the geodesic distance map. We therefore can solve the Eikonal equation to determine the distance between two points, and find the geodesic connecting the two points (by gradient descent).

In short, we relax the Boundary Value Problem (BVP)
```math
\begin{equation} \begin{dcases} \Vert \nabla_C W(p) \Vert_C = 1, & p \neq e, \\
W(p) = 0, & p = e, \end{dcases} \end{equation}
```
to a sequence of Initial Value Problem (IVP)
```math
\begin{equation} \begin{dcases} \partial_t W_{n + 1}(p, t) = 1 - \Vert \nabla_C W_{n + 1}(p, t) \Vert_C = 1, & t \in [0, 1], \\
W_{n + 1}(p, 0) = W_n(p, 1), & p \neq e, \\
W_{n + 1}(p, 0) = 0, & p = e. \end{dcases} \end{equation}
```
If we discretise in "time" sufficiently finely, and take sufficiently many steps, then the solution of the iterative IVP method (2) will in fact be the viscosity solution of the original BVP (1). The subscript $`C`$ in the equations above indicates that the operations are data-driven.

## Installation

1. Install [Miniforge](https://github.com/conda-forge/miniforge) (an alternative to Ana/Miniconda, with mamba built in).
2. Clone this repository.
3. In this directory, execute 
```
mamba env create -f=minenv.yml 
```

## Package Structure

### $`\mathbb{R}^2`$
We interpret $\mathbb{R}^2$ as a Riemannian manifold. The Eikonal PDE is then solved with respect to a metric that is data-driven left invariant under translations, which means that translating the input data will translate the output data correspondingly. The permitted data-driven metrics can be written as the product of a diagonal metric that is left invariant under translations with some cost function:
```math
\mathcal{G}^C|_{x, y} = C^2(x, y) \mathcal{G}^C|_{x, y} = C^2(x, y) g_i \mathrm{dx}^i|_{x, y} \otimes \mathrm{dx}^i|_{x, y} = C^2(x, y) (g_1 \mathrm{dx}|_{x, y} \otimes \mathrm{dx}|_{x, y} + g_2 \mathrm{dy}|_{x, y} \otimes \mathrm{dy}|_{x, y}),
```
where $`g_i`$ are constants. Here $`\mathrm{dx}`$ is the dual of $`\partial_x`$ and $`\mathrm{dy}`$ is the dual of $`\partial_y`$, where $`\{\partial_x, \partial_y\}`$ is the standard basis.

The implementation for $`\mathbb{R}^2`$ can be found in `eikivp/R2`.

### $`\mathbb{M}_2`$
We can also solve the Eikonal PDE on the space of positions and orientations on the plane,  $`\mathbb{M}_2 \coloneqq \mathbb{R}^2 \times S^1`$. $`\mathbb{M}_2`$ is naturally acted on by the Lie group $`\mathrm{SE}(2)`$ of roto-translations of the plane.* $`\mathrm{SE}(2)`$ induces left invariant vector fields on $`\mathbb{M}_2`$, which allow for roto-translation equivariant processing. We need to equip this smooth manifold with a notion of norms on tangent spaces before we can even consider the Eikonal equation. In this package, $`\mathbb{M}_2`$ can be equipped with three different types of norms: Riemannian, sub-Riemannian, and plus controller.

The implementation for $`\mathbb{M}_2`$ can be found in `eikivp/SE2`.

*_Indeed, $`\mathbb{M}_2`$ is the principal homogeneous space of $`\mathrm{SE}(2)`$. Upon choosing a reference position and orientation in $`\mathbb{M}_2`$, these two are isomorphic as smooth manifolds._

#### Riemannian
In this case, we see $`\mathbb{M}_2`$ as a Riemannian manifold. The Eikonal PDE is then solved with respect to metric that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven metrics can be written as the product of a diagonal metric that is left invariant under translations with some cost function:
```math
\begin{align*}\mathcal{G}^C|_{x, y, \theta} & = C^2(x, y, \theta) \mathcal{G}|_{x, y, \theta} = C^2(x, y, \theta) g_i \omega^i|_{x, y, \theta} \otimes \omega^i|_{x, y, \theta} \\
& = C^2(x, y, \theta) (g_1 \omega^1|_{x, y, \theta} \otimes \omega^1|_{x, y, \theta} + g_2 \omega^2|_{x, y, \theta} \otimes \omega^2|_{x, y, \theta} + g_3 \omega^3|_{x, y, \theta} \otimes \omega^3|_{x, y, \theta}),\end{align*}
```
where $`g_i`$ are constants. Here $\omega^i$ is the dual of $`\mathcal{A}_i`$, where $`\{\mathcal{A}_i\}`$ is the standard left invariant basis, given by*
```math
\begin{align*} \mathcal{A}_1|_{x, y, \theta} & = \cos(\theta) \partial_x|_{x, y, \theta} + \sin(\theta) \partial_y|_{x, y, \theta}, \\
\mathcal{A}_2|_{x, y, \theta} & = -\sin(\theta) \partial_x|_{x, y, \theta} + \cos(\theta) \partial_y|_{x, y, \theta}, \textrm{ and} \\
\mathcal{A}_3|_{x, y, \theta} & = \partial_{\theta}|_{x, y, \theta}.
\end{align*}
```

The implementation for $`\mathbb{M}_2`$ equipped with a Riemannian metric can be found in `eikivp/SE2/Riemannian`.

*_Note that we use the convention that_ $`\mathcal{A}_3 = \partial_{\theta}`$, _whereas Bekkers et al. say_ $`\mathcal{A}_2 = \partial_{\theta}`$.

#### Sub-Riemannian
In this case, we see $`\mathbb{M}_2`$ as a sub-Riemannian manifold. Specifically, we restrict the tangent spaces to be spanned by $`\mathcal{A}_1`$ and $`\mathcal{A}_3`$. This corresponds to the Reeds-Shepp car model [[2]](#2)[[3]](#3): the "car" is only allowed to move straight forward or backward, or turn the wheel.

The Eikonal PDE is then again solved with respect to a metric that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven metrics can be written as the product of a diagonal metric that is left invariant under translations with some cost function:
```math
\begin{align*}\mathcal{G}^C|_{x, y, \theta} & = C^2(x, y, \theta) \mathcal{G}|_{x, y, \theta} = C^2(x, y, \theta) g_i \omega^i|_{x, y, \theta} \otimes \omega^i|_{x, y, \theta} \\
& = C^2(x, y, \theta) (g_1 \omega^1|_{x, y, \theta} \otimes \omega^1|_{x, y, \theta} + g_3 \omega^3|_{x, y, \theta} \otimes \omega^3|_{x, y, \theta}),\end{align*}
```
where $`g_i`$ are constants. We always set $`g_3 = 1`$, and allow $`g_1 =: \xi^2`$ to be provided by the user (you can simply rescale the problem to effectively change $`g_3`$).

The implementation for $`\mathbb{M}_2`$ equipped with a sub-Riemannian metric can be found in `eikivp/SE2/subRiemannian`.

#### Plus controller
In this case, we see $`\mathbb{M}_2`$ as a Finslerian manifold. We again restrict the tangent spaces to be spanned by $`\mathcal{A}_1`$ and $`\mathcal{A}_3`$. Now, the car additionally is not allowed to move backwards [[3]](#3): this means motion in the negative $`\mathcal{A}_1`$ direction is restricted. Since Finslerian manifolds need not have a metric tensor corresponding to the Finsler function, and therefore also typically do not have a gradient $`\nabla`$, we need to reformulate the Eikonal PDE as follows:
```math
\begin{equation} \begin{dcases} (\mathcal{F}^C)^*(p, \mathrm{d}W(p)) = 1, & p \neq e, \\
W(p) = 0, & p = e, \end{dcases} \end{equation}
```
where $`\mathcal{F}^C`$ is the data-driven Finsler function, $`\cdot^*`$ denotes taking the dual, and $`\mathrm{d}W`$ is the differential of $`W`$. The IVP formulation can be likewise recast to work with Finslerian manifolds.

The Eikonal PDE is now solved with respect to a Finsler function that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven Finsler functions are so-called "plus controllers", which can be written as the product of a diagonal Finsler function that is left invariant under translations with some cost function:
```math
\begin{align*}(\mathcal{F}^C|_{x, y, \theta})^2 & = C^2(x, y, \theta) \mathcal{F}^2|_{x, y, \theta} = C^2(x, y, \theta) (g_1 (\omega^1|_{x, y, \theta})_+ \otimes (\omega^1|_{x, y, \theta})_+ + g_3 \omega^3|_{x, y, \theta} \otimes \omega^3|_{x, y, \theta}),\end{align*}
```
where $`g_i`$ are constants and $`(\cdot)_+`$ denotes taking the positive part. We always set $`g_3 = 1`$, and allow $`g_1 =: \xi^2`$ to be provided by the user (you can simply rescale the problem to effectively change $`g_3`$).

The implementation for $`\mathbb{M}_2`$ equipped with a Finsler function can be found in `eikivp/SE2/plus`.

### $`\mathbb{W}_2`$
We can also solve the Eikonal PDE on the space of positions and orientations on the sphere,  $`\mathbb{W}_2 \coloneqq S^2 \times S^1`$. $`\mathbb{W}_2`$ is naturally acted on by the Lie group $`\mathrm{SO}(3)`$ of rotations of the three dimensional Euclidean space.* $`\mathrm{SO}(3)`$ induces left invariant vector fields on $`\mathbb{W}_2`$, which allow for roto-translation equivariant processing. We need to equip this smooth manifold with a notion of norms on tangent spaces before we can even consider the Eikonal equation. In this package, $`\mathbb{W}_2`$ can be equipped with three different types of norms: Riemannian, sub-Riemannian, and plus controller.

We interpret $`\mathbb{W}_2`$ as a smooth manifold.* We need to equip this smooth manifold with a notion of norms on tangent spaces before we can even consider the Eikonal equation. In this package, $`\mathbb{W}_2`$ can be equipped with three different types of norms: Riemannian, sub-Riemannian, and plus controller.

The implementation for $`\mathbb{W}_2`$ can be found in `eikivp/SO3`.

*_Indeed, $`\mathbb{W}_2`$ is the principal homogeneous space of $`\mathrm{SO}(3)`$. Upon choosing a reference position and orientation in $`\mathbb{W}_2`$, these two are isomorphic as smooth manifolds._*

#### Riemannian
In this case, we see $`\mathbb{W}_2`$ as a Riemannian manifold. The Eikonal PDE is then solved with respect to metric that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven metrics can be written as the product of a diagonal metric that is left invariant under translations with some cost function:
```math
\begin{align*}\mathcal{G}^C|_{\alpha, \beta, \phi} & = C^2(\alpha, \beta, \phi) \mathcal{G}|_{\alpha, \beta, \phi} = C^2(\alpha, \beta, \phi) g_i \nu^i|_{\alpha, \beta, \phi} \otimes \nu^i|_{\alpha, \beta, \phi} \\
& = C^2(\alpha, \beta, \phi) (g_1 \nu^1|_{\alpha, \beta, \phi} \otimes \nu^1|_{\alpha, \beta, \phi} + g_2 \nu^2|_{\alpha, \beta, \phi} \otimes \nu^2|_{\alpha, \beta, \phi} + g_3 \nu^3|_{\alpha, \beta, \phi} \otimes \nu^3|_{\alpha, \beta, \phi}),\end{align*}
```
where $`g_i`$ are constants. Here $\nu^i$ is the dual of $`\mathcal{B}_i`$, where $`\{\mathcal{B}_i\}`$ is the standard left invariant basis, given by
```math
\begin{align*} \mathcal{B}_1|_{\alpha, \beta, \phi} & = \cos(\phi) \partial_{\alpha}|_{\alpha, \beta, \phi} + \frac{\sin(\phi)}{\cos(\alpha)} \partial_{\beta}|_{\alpha, \beta, \phi} + \sin(\phi) \tan(\alpha) \partial_{\phi}|_{\alpha, \beta, \phi}, \\
\mathcal{B}_2|_{\alpha, \beta, \phi} & = -\sin(\phi) \partial_{\alpha}|_{\alpha, \beta, \phi} + \frac{\cos(\phi)}{\cos(\alpha)} \partial_{\beta}|_{\alpha, \beta, \phi} + \cos(\phi) \tan(\alpha) \partial_{\phi}|_{\alpha, \beta, \phi}, \textrm{ and} \\
\mathcal{B}_3|_{\alpha, \beta, \phi} & = \partial_{\phi}|_{\alpha, \beta, \phi}.
\end{align*}
```

The implementation for $`\mathbb{W}_2`$ equipped with a Riemannian metric can be found in `eikivp/SO3/Riemannian`.

#### Sub-Riemannian
In this case, we see $`\mathbb{W}_2`$ as a sub-Riemannian manifold. Specifically, we restrict the tangent spaces to be spanned by $`\mathcal{B}_1`$ and $`\mathcal{B}_3`$. This corresponds to the Reeds-Shepp car model [[2]](#2)[[3]](#3): the "car" is only allowed to move straight forward or backward, or turn the wheel.

The Eikonal PDE is then again solved with respect to a metric that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven metrics can be written as the product of a diagonal metric that is left invariant under translations with some cost function:
```math
\begin{align*}\mathcal{G}^C|_{\alpha, \beta, \phi} & = C^2(\alpha, \beta, \phi) \mathcal{G}|_{\alpha, \beta, \phi} = C^2(\alpha, \beta, \phi) g_i \nu^i|_{\alpha, \beta, \phi} \otimes \nu^i|_{\alpha, \beta, \phi} \\
& = C^2(\alpha, \beta, \phi) (g_1 \nu^1|_{\alpha, \beta, \phi} \otimes \nu^1|_{\alpha, \beta, \phi} + g_3 \nu^3|_{\alpha, \beta, \phi} \otimes \nu^3|_{\alpha, \beta, \phi}),\end{align*}
```
where $`g_i`$ are constants. We always set $`g_3 = 1`$, and allow $`g_1 =: \xi^2`$ to be provided by the user (you can simply rescale the problem to effectively change $`g_3`$).

The implementation for $`\mathbb{W}_2`$ equipped with a sub-Riemannian metric can be found in `eikivp/SO3/subRiemannian`.

#### Plus controller
In this case, we see $`\mathbb{W}_2`$ as a Finslerian manifold. We again restrict the tangent spaces to be spanned by $`\mathcal{B}_1`$ and $`\mathcal{B}_3`$. Now, the car additionally is not allowed to move backwards [[3]](#3): this means motion in the negative $`\mathcal{B}_1`$ direction is restricted. Since Finslerian manifolds need not have a metric tensor corresponding to the Finsler function, and therefore also typically do not have a gradient $`\nabla`$, we need to reformulate the Eikonal PDE as follows:
```math
\begin{equation} \begin{dcases} (\mathcal{F}^C)^*(p, \mathrm{d}W(p)) = 1, & p \neq e, \\
W(p) = 0, & p = e, \end{dcases} \end{equation}
```
where $`\mathcal{F}^C`$ is the data-driven Finsler function, $`\cdot^*`$ denotes taking the dual, and $`\mathrm{d}W`$ is the differential of $`W`$. The IVP formulation can be likewise recast to work with Finslerian manifolds.

The Eikonal PDE is now solved with respect to a Finsler function that is data-driven left invariant under translations and rotations, which means that translating and rotating the input data will translate and rotate the output data correspondingly. The permitted data-driven Finsler functions are so-called "plus controllers", which can be written as the product of a diagonal Finsler function that is left invariant under translations with some cost function:
```math
\begin{align*}(\mathcal{F}^C|_{\alpha, \beta, \phi})^2 & = C^2(\alpha, \beta, \phi) \mathcal{F}^2|_{\alpha, \beta, \phi} = C^2(\alpha, \beta, \phi) (g_1 (\nu^1|_{\alpha, \beta, \phi})_+ \otimes (\nu^1|_{\alpha, \beta, \phi})_+ + g_3 \nu^3|_{\alpha, \beta, \phi} \otimes \nu^3|_{\alpha, \beta, \phi}),\end{align*}
```
where $`g_i`$ are constants and $`(\cdot)_+`$ denotes taking the positive part. We always set $`g_3 = 1`$, and allow $`g_1 =: \xi^2`$ to be provided by the user (you can simply rescale the problem to effectively change $`g_3`$).

The implementation for $`\mathbb{W}_2`$ equipped with a Finsler function can be found in `eikivp/SO3/plus`.

## References
<a id="1">[1]</a> 
E.J. Bekkers, R. Duits, A. Mashtakov, and G.R. Sanguinetti (2015) 
["A PDE Approach to Data-Driven Sub-Riemannian Geodesics in $SE(2)$"](https://epubs.siam.org/doi/abs/10.1137/15M1018460).

<a id="2">[2]</a> 
J.A. Reeds, III, and L.A. Shepp (1990) 
["Optimal paths for a car that goes both forwards and backwards"](https://msp.org/pjm/1990/145-2/p06.xhtml).

<a id="3">[3]</a> 
R. Duits, S.P.L. Meesters, J.-M. Mirebeau, and J.M. Portegies (2018) 
["Optimal Paths for Variants of the 2D and 3D Reeds-Shepp Car with Applications in Image Analysis"](https://link.springer.com/article/10.1007/s10851-018-0795-z).
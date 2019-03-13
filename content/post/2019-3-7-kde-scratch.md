---
title: Kernel Density Estimation from Scratch
author: J. Duncan
date: '2019-03-08'
tags:
  - julia
mathjax: yes
options:
  fig_ext: .svg
---

## KDE with KernelDensity.jl

From the Kernel Density Estimation Wikipedia [site](https://en.wikipedia.org/wiki/Kernel_density_estimation) we can take a look at reproducing the Example section.

```julia
using StatsPlots

x = [-2.1; -1.3; -0.4; 1.9; 5.1; 6.2];
scatter(x, zeros(length(x)), legend = false)
```

![](/post/2019-3-7-kde-scratch_files/2019-3-7-kde-scratch_1_1.svg)

Now applying the quick and easy solution: a package.

```julia
import KernelDensity

KernelDensity.kde(x, bandwidth = sqrt(2.25)) |>
  x -> plot!(x, legend = false)
```


![](/post/2019-3-7-kde-scratch_files/2019-3-7-kde-scratch_2_1.svg)


## KDE with Distributions.jl

Creating individual distributions for each data point.

````julia
using Distributions

dists = Normal.(x, sqrt(2.25))
````


````
6-element Array{Distributions.Normal{Float64},1}:
 Distributions.Normal{Float64}(μ=-2.1, σ=1.5)
 Distributions.Normal{Float64}(μ=-1.3, σ=1.5)
 Distributions.Normal{Float64}(μ=-0.4, σ=1.5)
 Distributions.Normal{Float64}(μ=1.9, σ=1.5)
 Distributions.Normal{Float64}(μ=5.1, σ=1.5)
 Distributions.Normal{Float64}(μ=6.2, σ=1.5)
````

The `Normal` function above is vectorized by the use of the `.` operator. This creates an array of Normal distributions. Let's plot each of these distributions:

````julia
plot(dists, legend = false)
````

![](/post/2019-3-7-kde-scratch_files/2019-3-7-kde-scratch_4_1.svg)


Summing up their probability densities across all of `x`.

````julia
x_d = range(-7, 11, length = 100)
dens = sum(pdf.(eachdist, x_d) for eachdist in dists)

plot!(x_d, dens)
````

![](/post/2019-3-7-kde-scratch_files/2019-3-7-kde-scratch_5_1.svg)


Interject a little math:

`$$
\hat f_h(x) = \frac{1}{n} \sum_{i=1}^n K_h(x - x_i)
$$`


## Kernel Density from Scratch

````julia
kde(z, ω, xv) =
  sum(@. exp(-0.5 * ((z - xv) / ω)^2) / sqrt(2π*ω^2)) / length(xv)

dens = []
for x_d in range(-7, 10, length = 100)
  push!(dens, kde(x_d, sqrt(2.25), x))
end

plot(dens, legend = false)
````

![](/post/2019-3-7-kde-scratch_files/2019-3-7-kde-scratch_6_1.svg)

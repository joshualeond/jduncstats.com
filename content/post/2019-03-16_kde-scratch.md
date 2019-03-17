---
title : Kernel Density Estimation from Scratch
author : J. Duncan
date: 2019-03-16T00:00:00-05:00
tags:
  - julia
options:
  fig_ext: ".svg"
---

## Motivation
Why am I writing about KDE's? At the recommendation of Will Kurt's [probability blog](https://www.countbayesie.com/) I've been reading a great book by Phillip K. Jannert:

<img src="/post/2019-03-16-kde-scratch_files/janert.jpg" alt="drawing" width="50%"/>

This book has a great overview of KDE's that explain the motivation. My own abbreviated version is that KDE's are a visualization technique to view a variables distribution. It's best to visualize a variable prior to calculating measures of central tendency (means/modes/medians) and other statistics like variance and standard deviation. It's important because these measures are assuming that your data is `unimodal` (having a single peak). If your data doesn't have this structure then these measures may be misleading.

This post follows closely with how I commonly learn about topics. I start at a high level, using a pre-canned solution for the algorithm, and then work backward to find out what's going on underneath.

I use a small example I discovered on the [Wikipedia page](https://en.wikipedia.org/wiki/Kernel_density_estimation)) for KDEs. It uses a handful of data points for a single variable:

Sample | Value
------ | -----
 1 | -2.1
 2 | -1.3
 3 | -0.4
 4 | 1.9
 5 | 5.1
 6 | 6.2

I'll be using the `Julia Programming Language` for this example. Let's start with a basic `dot` plot of these points.

````julia
using StatsPlots

# initialize an array of the samples
x = [-2.1; -1.3; -0.4; 1.9; 5.1; 6.2];
# plot it out
scatter(x, zeros(length(x)), legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_1_1.svg)



## KDE with KernelDensity.jl

Now applying the quick and easy solution: a [package](https://github.com/JuliaStats/KernelDensity.jl). This package has a function named `kde` that takes a one dimensional array (or vector), a bandwidth argument, and a chosen kernel (we'll use the default). So let's see it:

````julia
import KernelDensity

KernelDensity.kde(x, bandwidth = sqrt(2.25)) |>
  x -> plot!(x, legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_2_1.svg)



There we go, we've applied KDE to these data points and we can see the `bimodal` nature of this data. We could stop here but let's keep going.

## KDE with Distributions.jl

What is the `kernel` part of this about? The `kde` function from the package uses a default `kernel` associated with the `Normal` distribution. But to understand what this all means we need to take a look at the definition of Kernel Density Estimation:

`$$
D_h(x; {x_i}) = \sum_{i=1}^n \frac{1}{h} K(\frac{x - x_i}{h})
$$`

Breaking down this formula a bit: The `kernel` is the function shown above as `$K$`. Jannert describes it as a "highly" peaked function. For every point along the x-axis we take the distance between that point and an individual data point from our example. That set of distances are fed into the kernel function. Then the results of these evaluations for each individual data point are summed together and divided by the number of observations of our variable.

There is a great demonstration of how kernel functions work [here](https://mathisonian.github.io/kde/). It is a very elegant and interactive interface where you can test different bandwidths/kernels and see the results of the KDE.

As I mentioned before, the default `kernel` for this package is the `Normal` (or Gaussian) probability density function (pdf):

`$$
K(x) = \frac{1}{\sqrt{2\pi}}\text{exp}(-\frac{1}{2}x^2)
$$`

We can get an idea of what the KDE should look like by using the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package. We can create individual normal distributions for each data point and then calculate the `pdf` of each distribution across the x-axis and sum these results.

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





Here we see a neat feature of the Julia language. Any Julia function can be vectorized. See [this](https://julialang.org/blog/2018/05/extensible-broadcast-fusion) blog post if you want to learn more about this feature. The `Normal` function above is vectorized by the use of the `.` operator. This creates an array of Normal distributions. Let's plot each of these distributions:

````julia
plot(dists, legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_4_1.svg)



Summing up their probability densities across all of `x`.

````julia
x_d = range(-7, 11, length = 100)
dens = sum(pdf.(eachdist, x_d) for eachdist in dists)

plot!(x_d, dens)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_5_1.svg)



The resulting shape of the KDE is identical to the one we first calculated. We could stop here except this is really just a special case where we are using the Normal kernel. Let's go a bit deeper.

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


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_6_1.svg)

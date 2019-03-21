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
Why am I writing about KDE's? At the recommendation of Will Kurt's [probability blog](https://www.countbayesie.com/) I've been reading a great book on data analysis by Philipp K. Janert:

<img src="/post/2019-03-16-kde-scratch_files/janert.jpg" alt="drawing" width="50%"/>

This book has a great intro to KDEs that explain the motivation. My own abbreviated version are that KDEs provide a useful technique to visualize a variables distribution. Visualizing your data is an important step to take early in the data analysis stage. In fact, there are many metrics that we commonly use to understand a variable that have an implicit assumption that your data are `unimodal` (having a single peak). If your data doesn't have this structure then you may be mislead by measures of central tendency (mean/median/mode), outliers, or other statistical methods (linear regression, t-tests, etc.).

This posts structure follows closely with how I commonly learn topics. I start at a high level, using a pre-canned solution for the algorithm, and then work backward to find out what's going on underneath.

I use a small example I discovered on the [Wikipedia page](https://en.wikipedia.org/wiki/Kernel_density_estimation) for KDEs. It uses a handful of data points for a single variable:

Sample | Value
------ | -----
 1 | -2.1
 2 | -1.3
 3 | -0.4
 4 | 1.9
 5 | 5.1
 6 | 6.2

Let's start with a basic `dot` plot of these points. I'll be using the `Julia Programming Language` for these examples.

````julia
using StatsPlots

# initialize an array of the samples
x = [-2.1; -1.3; -0.4; 1.9; 5.1; 6.2];
# plot it out
scatter(x, zeros(length(x)), legend = false)
````


![](/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_1_1.svg)


## KDE with KernelDensity.jl

Now applying the quick and easy solution: a [package](https://github.com/JuliaStats/KernelDensity.jl). This package has a function named `kde` that takes a one dimensional array (or vector), a bandwidth argument, and a chosen kernel (we'll use the default). So let's see it:

````julia
import KernelDensity

KernelDensity.kde(x, bandwidth = sqrt(2.25)) |>
  x -> plot!(x, legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_2_1.svg)



There we go, we've applied KDE to these data points and we can now see the `bimodal` nature of this data. If all we wanted to do was visualize the distribution then we're done. I'd like to dig a bit deeper though.

## KDE with Distributions.jl

What is the `kernel` part of this about? What was the default kernel we used in the previous section? The `kde` function from the package used a default `kernel` associated with the `Normal` distribution. But to understand what this all means we need to take a look at the definition of Kernel Density Estimation:

`$$
D_h(x; {x_i}) = \sum_{i=1}^n \frac{1}{nh} K\left(\frac{x - x_i}{h}\right)
$$`

Breaking down this formula a bit: The `kernel` is the function shown above as `$K$` and Janert describes it like so:

> To form a KDE, we place a *kernel* —that is, a smooth, strongly peaked function—at the
position of each data point. We then add up the contributions from all kernels to obtain a
smooth curve, which we can evaluate at any point along the *x* axis

We are effectively calculating weighted distances from our data points to points along the *x* axis. There is a great interactive introduction to kernel density estimation [here](https://mathisonian.github.io/kde/). I highly recommend it because you can play with bandwidth, select different kernel methods, and check out the resulting effects.

As I mentioned before, the default `kernel` for this package is the `Normal` (or Gaussian) probability density function (pdf):

`$$
K(x) = \frac{1}{\sqrt{2\pi}}\text{exp}\left(-\frac{1}{2}x^2\right)
$$`

Since we are calculating `pdf`s I'll use the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package to create each distribution, calculate the densities, and sum the results.

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





Here we see a neat feature of the Julia language. Any Julia function can be vectorized (or broadcasted) by the application of the `.` (or "dot") operator. See [this](https://julialang.org/blog/2018/05/extensible-broadcast-fusion) blog post if you want to learn more about it. Above we applied the `Normal` method element-wise creating an array of Normal distributions. The mean of our individual distributions being our data points and the standard deviation being our chosen bandwidth (`2.25`). Let's plot each of these distributions:

````julia
plot(dists, legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_4_1.svg)



Summing up their probability densities across all of `x`.

````julia
# create an iterator
x_d = range(-7, 11, length = 100)
# find the kde with a gaussian kernel
dens = sum(pdf.(eachdist, x_d) for eachdist in dists)

plot!(x_d, dens)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_5_1.svg)



The resulting shape of the KDE is identical to the one we first calculated. We could stop here except this is really just a special case where we are using the gaussian kernel. Let's extrapolate a bit so we could use different kernels.

## Kernel Density from Scratch

To apply a new kernel method we can just write the KDE code from scratch. Below I've defined the gaussian kernel as `K(x)` and the KDE function as `D(x)` to mimic the math above.

````julia
# gaussian kernel
K(x) = 1/sqrt(2π) * exp(-1/2 * x^2)

# define the KDE function
D(x, h, x_i) =
  1/(length(x_i) * h) * sum(K.((x .- x_i) / h))

# evaluate KDE along the x-axis
dens = []
for xstep in x_d
  push!(dens, D(xstep, sqrt(2.25), x))
end

plot(x_d, dens, legend = false)
````


![](/post/2019-03-16-kde-scratch_files/2019-03-16_kde-scratch_6_1.svg)



I've defined these functions using Julia's "assignment form". I did this because of the compact nature of this particular implementation. To generalize the kernel function, `D(x)`, to be close to the `kde` function that we used in the first step we would want to use the traditional function definition like so:

````julia
function foo(x)
  ...
  return y
end
````

Then we could add a `kernel` argument to our function, `D(x)` or declare [types](https://docs.julialang.org/en/v1/manual/performance-tips/#Type-declarations-1) for cleaner code.

## What's next?

A short post on cumulative distribution functions (`cdf`) using Julia will likely follow this one. Janert introduces both `kde`s and `cdf`s in his chapter **A Single Variable: Shape and Distribution** and they complement each other really well. Thanks for reading!

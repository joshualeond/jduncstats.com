---
title : Model building of golf putting with Turing.jl
author : J. Duncan
date: 2019-11-02T00:00:00-05:00
image:
  focal_point: ""
  preview_only: false
tags:
  - julia
  - bayes
  - Turing.jl
options:
  fig_ext: ".svg"
  doctype :  github
---

## First Thing (Disclaimer)

This blog post is based on a blog post written by the popular bayesian statistician Andrew Gelman.

Dr. Gelman's post was written in the [Stan](https://mc-stan.org/) probabilistic programming language (or PPL). Since this post, there have been a couple other blog posts translating the code to other PPLs. They are excellent and I definitely recommend checking them out. Here's the list of links and my opinion on some of the benefits of reading each:

* Dr. Gelman's using Stan with R: See [here](https://mc-stan.org/users/documentation/case-studies/golf.html). Read Dr. Gelman's post first as he lays out the problem being modeled here really well. Explaining the geometry at play and how to adjust the models to incorporate the physics of the problem. It's a great article showing the power of using a probabilistic programming language to build models.
* Colin Carroll's using the Python PPL PyMC3: You can check out his work [here](https://nbviewer.jupyter.org/github/pymc-devs/pymc3/blob/master/docs/source/notebooks/putting_workflow.ipynb). I really enjoyed Colin's post because of his prior predictive checks, data visualizations, and showing the value of these models moving past predictions.
* Adam Haber's post using Python and Tensorflow Probability: See [here](https://adamhaber.github.io/post/nuts/). His post goes into great detail about more of the sampling and lower level details of Tensorflow Probability and MCMC.

My post is nothing really novel, simply a port to the Julia PPL, [Turing.jl](https://turing.ml/dev/). This post shows how to specify and estimate these same models within the Julia programming language.

## Getting started and plotting

Let's Load the data and take a look:

````julia
using Turing, CSV
# Hide sampling progress.
Turing.turnprogress(false);

data = CSV.read("golf_data.csv")
x, n, y = (data[:,1], data[:,2], data[:,3])
````

We have three variables in this dataset:

Variable | Units | Description
------ | ----- | -----
x  | feet | The distance of the attempted putt
n | count | The total attempts (or trials) of putts at a chosen distance
y | count | The total successful putts from the total attempts

What we are attempting to build is a model to predict the probability of success given the distance from the hole. We need to transform this data a bit to explore the dataset visually. Let's calculate the probabilities and the error involved. We'll use the following formula to calculate the error bars for the putting success rate:

$$
\sqrt{\hat{p}_j(1-\hat{p}_j)/n_j}
$$

````julia
pj = y ./ n
error = @. sqrt((pj * (1 - pj) / n));
````

Now let's visualize the dataset:

````julia
using Plots

# plot out the error
scatter(
  x, pj,
  yerror= error,
  legend = false,
  ylim = (0, 1),
  ylab = "Probability of Success",
  xlab = "Distance from hole (ft)")
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_2_1.svg)



## Logistic Regression

Building our first model, a GLM (Generalized Linear Model). We will attempt to model the probability of success in golf putting incorporating the distance as an independent (or predictor) variable.

$$
y_j\sim\mbox{binomial}(n_j, \mbox{logit}^{-1}(a + bx_j)),
\mbox{ for } j=1,\dots, J.
$$


````julia
using StatsFuns: logistic

@model golf_logistic(x,y,n,J) = begin
  # parameters
  a ~ Normal(0, 1)
  b ~ Normal(0, 1)

  # model
  for i in 1:J
    p = logistic(a + b * x[i])
    y[i] ~ Binomial(n[i], p)
  end
end

chn = psample(golf_logistic(x, y, n, length(x)), NUTS(), 4000, 2);
````


I'd like to add here that I'm using the `psample` method in `Turing.jl` to sample this model. If you have read through the documentation for `Turing.jl` you'll have seen the `sample` method used more often. The only difference here is that the `psample` method performs parallel sampling (using the [new](https://julialang.org/blog/2019/07/multithreading) multi-threaded functionality available in Julia version `1.3`) and thus gives us more than one chain. That last argument in the function call is the number of chains I'd like to use. You can read more about it in the [documentation](https://turing.ml/dev/docs/using-turing/guide#sampling-multiple-chains).

Now we've sampled the joint probability distribution. Let's take a look at the results. I'll create a function to show the table of results as html using the `PrettyTables.jl` package. Looking at the summary statistics for the posterior distribution:

````julia
using PrettyTables

function prettystats(chains)
  chains |>
    x -> summarystats(x, digits = 2) |>
    x -> x.df |>
    x -> pretty_table(x, backend = :html)
end

prettystats(chn)
````


<html>
<meta charset="UTF-8">
<style>
table, td, th {
    border-collapse: collapse;
    font-family: sans-serif;
}

td, th {
    border-bottom: 0;
    padding: 4px
}

tr:nth-child(odd) {
    background: #eee;
}

tr:nth-child(even) {
    background: #fff;
}

tr.header {
    background: navy !important;
    color: white;
    font-weight: bold;
}

tr.subheader {
    background: lightgray !important;
    color: black;
}

tr.headerLastRow {
    border-bottom: 2px solid black;
}

th.rowNumber, td.rowNumber {
    text-align: right;
}

</style>
<body>
<table>

<tr class = header>
<th style = "text-align: right; ">parameters</th>
<th style = "text-align: right; ">mean</th>
<th style = "text-align: right; ">std</th>
<th style = "text-align: right; ">naive_se</th>
<th style = "text-align: right; ">mcse</th>
<th style = "text-align: right; ">ess</th>
<th style = "text-align: right; ">r_hat</th>
</tr>
<tr class = "subheader headerLastRow">
<th style = "text-align: right; ">Symbol</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Any</th>
<th style = "text-align: right; ">Any</th>
</tr>
<tr>
<td style = "text-align: right; ">a</td>
<td style = "text-align: right; ">2.23</td>
<td style = "text-align: right; ">0.06</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">899.26</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">b</td>
<td style = "text-align: right; ">-0.26</td>
<td style = "text-align: right; ">0.01</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">1173.64</td>
<td style = "text-align: right; ">1.0</td>
</tr>
</table></body></html>



Visualizing the predictions:

````julia
a_post = median(chn[:a].value)
b_post = median(chn[:b].value)

# iterator for distance from hole calcs
xrng = 1:1:21
post_lines = [logistic(a_post + b_post * x) for x = xrng]

# 50 draws from the posterior
using StatsBase
a_samp = StatsBase.sample(chn[:a].value, 50)
b_samp = StatsBase.sample(chn[:b].value, 50)

post_samp = [logistic(a_samp[i] + b_samp[i] * x) for x = xrng, i = 1:50]

plot!(post_samp, alpha = 0.5, color = :gray) # add uncertainty samples
plot!(post_lines, color = :black) # add median
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_5_1.svg)



## First Principles

The next step is building a more bespoke model that incorporates the physics of this problem. Dr. Gelman describes this step as follows:

> We assume that the golfer is attempting to hit the ball completely straight but that many small factors interfere with this goal, so that the actual angle follows a normal distribution centered at 0 with some standard deviation σ.

> The probability the ball goes in the hole is then the probability that the angle is less than the threshold; that is $$
\mbox{Pr}\left(|\mbox{angle}| < \sin^{-1}((R-r)/x)\right) = 2\Phi\left(\frac{\sin^{-1}((R-r)/x)}{\sigma}\right) - 1
$$

> where Φ is the cumulative normal distribution function.

Again, for more background I would suggest reading the original post. So now we'll define the function `Phi`. It's the cumulative distribution function of the standard normal distribution.

````julia
Phi(x) = cdf.(Normal(0, 1), x);
````



Now let's create and sample this model incorporating the angle.

````julia
@model golf_angle(x, y, n, J, r, R) = begin
  # transformed data
  threshold_angle = asin.((R - r) ./ x)

  # parameters
  sigma ~ Truncated(Normal(0, 1), 0, Inf)

  # model
  p = 2 * Phi(threshold_angle / sigma) .- 1
  for i in 1:J
    y[i] ~ Binomial(n[i], p[i])
  end
end

# radius of ball and hole respectively
r = (1.68 / 2) / 12
R = (4.25 / 2) / 12

chn2 = psample(golf_angle(x, y, n, length(x), r, R), NUTS(), 4000, 2)
chn2 = hcat(chn2, Chains(chn2[:sigma].value * 180 / π, ["sigma_degrees"]))

prettystats(chn2)
````


<html>
<meta charset="UTF-8">
<style>
table, td, th {
    border-collapse: collapse;
    font-family: sans-serif;
}

td, th {
    border-bottom: 0;
    padding: 4px
}

tr:nth-child(odd) {
    background: #eee;
}

tr:nth-child(even) {
    background: #fff;
}

tr.header {
    background: navy !important;
    color: white;
    font-weight: bold;
}

tr.subheader {
    background: lightgray !important;
    color: black;
}

tr.headerLastRow {
    border-bottom: 2px solid black;
}

th.rowNumber, td.rowNumber {
    text-align: right;
}

</style>
<body>
<table>

<tr class = header>
<th style = "text-align: right; ">parameters</th>
<th style = "text-align: right; ">mean</th>
<th style = "text-align: right; ">std</th>
<th style = "text-align: right; ">naive_se</th>
<th style = "text-align: right; ">mcse</th>
<th style = "text-align: right; ">ess</th>
<th style = "text-align: right; ">r_hat</th>
</tr>
<tr class = "subheader headerLastRow">
<th style = "text-align: right; ">Symbol</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Any</th>
<th style = "text-align: right; ">Any</th>
</tr>
<tr>
<td style = "text-align: right; ">sigma</td>
<td style = "text-align: right; ">0.03</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4414.07</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">sigma_degrees</td>
<td style = "text-align: right; ">1.53</td>
<td style = "text-align: right; ">0.02</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4414.07</td>
<td style = "text-align: right; ">1.0</td>
</tr>
</table></body></html>



Now we can calculate predictions and see how this model compares to the logistic model. Let's wrap the angle calculation into a function as we'll use it frequently throughout the post.

````julia
prob_angle(threshold, sigma) = 2 * Phi(threshold / sigma) .- 1;
````

Calculate and visualize predictions.

````julia
# calculate predictions
post_sigma = median(chn2[:sigma].value)
threshold_angle = [asin((R - r) / x) for x = xrng]
geom_lines = prob_angle(threshold_angle, post_sigma)

scatter(
  x, pj,
  yerror= error,
  label = "",
  ylim = (0, 1),
  ylab = "Probability of Success",
  xlab = "Distance from hole (ft)")
plot!(post_lines, color = :black, label = "Logistic regression")
plot!(geom_lines, color = 1, label = "Geometry-based model")
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_9_1.svg)

We see that the geometry based model fits the data much better than the Logistic regression.

## New Golf Data

Then came new data. Dr. Gelman received new data of golf putting and compared the fit of the original geometry based model with the old and new data. Here is the comparison below:

````julia
datanew = CSV.read("golf_data_new.csv")
xnew, nnew, ynew = (datanew[:,1], datanew[:,2], datanew[:,3])
pnew = ynew ./ nnew
xrngnew = 1:1:80

# plot the old model fit with new data
threshold_angle2 = [asin((R - r) / x) for x = xrngnew]
geom_lines2 = prob_angle(threshold_angle2, post_sigma)

scatter(
  x, pj,
  label = "Old data",
  ylab = "Probability of Success",
  xlab = "Distance from hole (ft)",
  color = 1)
scatter!(xnew, pnew, color = 2, label = "New data")
plot!(geom_lines2, label = "", color = 1)
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_10_1.svg)

We see that the new data have many more observations with longer distance putts than in the original data. The probability of success for these longer distance putts do not agree with the original geometry based model.

## Updated Geometry

So Dr. Gelman improves the model by taking distance into account.

> To get the ball in the hole, the angle isn’t the only thing you need to control; you also need to hit the ball just hard enough.

> ...the probability a shot goes in becomes, $$
\left(2\Phi\left(\frac{\sin^{-1}((R-r)/x)}{\sigma_{\rm angle}}\right) - 1\right)\left(\Phi\left(\frac{2}{(x+1)\,\sigma_{\rm distance}}\right) - \Phi\left(\frac{-1}{(x+1)\,\sigma_{\rm distance}}\right)\right)
$$

> where we have renamed the parameter σ from our earlier model to σ_angle to distinguish it from the new σ_distance parameter.

Let's add a function for the distance calculation for the improved geometry.

````julia
prob_distance(distance, tol, overshot, sigma) =
  Phi((tol - overshot) ./ ((distance .+ overshot) * sigma)) -
    Phi(-overshot ./ ((distance .+ overshot) * sigma));
````

Now let's create the model to sample:

````julia
@model golf_angle_dist(x, y, n, J, r, R, overshot, distance_tolerance) = begin
  # transformed data
  threshold_angle = asin.((R - r) ./ x)

  # parameters
  sigma_angle ~ Truncated(Normal(0, 1), 0, Inf)
  sigma_distance ~ Truncated(Normal(0, 1), 0, Inf)

  # model
  p_angle = prob_angle(threshold_angle, sigma_angle)
  p_distance = prob_distance(x, distance_tolerance, overshot, sigma_distance)
  p = p_angle .* p_distance

  for i in 1:J
    y[i] ~ Binomial(n[i], p[i])
  end
end

overshot = 1.
distance_tolerance = 3.

chn3 = psample(
  golf_angle_dist(xnew, ynew, nnew, length(xnew), r, R, overshot, distance_tolerance),
  NUTS(),
  8000,
  2)
prettystats(chn3)
````


<html>
<meta charset="UTF-8">
<style>
table, td, th {
    border-collapse: collapse;
    font-family: sans-serif;
}

td, th {
    border-bottom: 0;
    padding: 4px
}

tr:nth-child(odd) {
    background: #eee;
}

tr:nth-child(even) {
    background: #fff;
}

tr.header {
    background: navy !important;
    color: white;
    font-weight: bold;
}

tr.subheader {
    background: lightgray !important;
    color: black;
}

tr.headerLastRow {
    border-bottom: 2px solid black;
}

th.rowNumber, td.rowNumber {
    text-align: right;
}

</style>
<body>
<table>

<tr class = header>
<th style = "text-align: right; ">parameters</th>
<th style = "text-align: right; ">mean</th>
<th style = "text-align: right; ">std</th>
<th style = "text-align: right; ">naive_se</th>
<th style = "text-align: right; ">mcse</th>
<th style = "text-align: right; ">ess</th>
<th style = "text-align: right; ">r_hat</th>
</tr>
<tr class = "subheader headerLastRow">
<th style = "text-align: right; ">Symbol</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Any</th>
<th style = "text-align: right; ">Any</th>
</tr>
<tr>
<td style = "text-align: right; ">sigma_angle</td>
<td style = "text-align: right; ">0.01</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4092.49</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">sigma_distance</td>
<td style = "text-align: right; ">0.14</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4348.54</td>
<td style = "text-align: right; ">1.0</td>
</tr>
</table></body></html>




Dr. Gelman's post suggests something is unstable with this model estimation. The estimated parameters match his closely however we do have drastically different values for the `$\hat R$` diagnostics. I did notice that sampling this model did not always give me same results though. I ended up increasing the length of the chains to try to get around the inconsistent parameter estimation I was experiencing.

Now let's make some predictions and visualize the results:

````julia
# calculate predictions
post_siga = median(chn3[:sigma_angle].value)
post_sigd = median(chn3[:sigma_distance].value)

p_angle = prob_angle(threshold_angle2, post_siga)
p_distance = prob_distance(xrngnew, distance_tolerance, overshot, post_sigd)

geom2_lines = p_angle .* p_distance

# plot
scatter(
  xnew, pnew,
  legend = false,
  color = 2,
  ylab = "Probability of Success",
  xlab = "Distance from hole (ft)")
plot!(geom2_lines, color = 2)
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_13_1.svg)

Now this model fits the new data better than the original geometry based model but you can see an issue near the middle of the range of the x-axis (distance). Some of Gelman's select comments and proposed fix:

> There are problems with the fit in the middle of the range of x. We suspect this is a problem with the binomial error model, as it tries harder to fit points where the counts are higher. Look at how closely the fitted curve hugs the data at the very lowest values of x.

> To fix this problem we took the data model, $ y_j \sim \mbox{binomial}(n_j, p_j) $, and added an independent error term to each observation.

> ...we first approximate the binomial data distribution by a normal and then add independent variance; thus: $$ y_j/n_j \sim \mbox{normal}\left(p_j, \sqrt{p_j(1-p_j)/n_j + \sigma_y^2}\right) $$

## A Dispersed Model

Now let's implement the changes referenced above within Turing:

````julia
@model golf_angle_dist_resid(x, y, n, J, r, R, overshot, distance_tolerance, raw) = begin
  # transformed data
  threshold_angle = asin.((R - r) ./ x)

  # parameters
  sigma_angle ~ Truncated(Normal(0, 1), 0, Inf)
  sigma_distance ~ Truncated(Normal(0, 1), 0, Inf)
  sigma_y ~ Truncated(Normal(0, 1), 0, Inf)

  # model
  p_angle = prob_angle(threshold_angle, sigma_angle)
  p_distance = prob_distance(x, distance_tolerance, overshot, sigma_distance)
  p = p_angle .* p_distance

  for i in 1:J
    raw[i] ~ Normal(p[i], sqrt(p[i] * (1-p[i]) / n[i] + sigma_y^2))
  end
end

chn4 = psample(
  golf_angle_dist_resid(xnew, ynew, nnew, length(xnew), r, R, overshot, distance_tolerance, ynew ./ nnew),
  NUTS(),
  4000,
  2)

# adding the conversion to degrees
chns = hcat(chn4, Chains(chn4[:sigma_angle].value * 180 / π, ["sigma_degrees"]))
prettystats(chns)
````

<html>
<meta charset="UTF-8">
<style>
table, td, th {
    border-collapse: collapse;
    font-family: sans-serif;
}

td, th {
    border-bottom: 0;
    padding: 4px
}

tr:nth-child(odd) {
    background: #eee;
}

tr:nth-child(even) {
    background: #fff;
}

tr.header {
    background: navy !important;
    color: white;
    font-weight: bold;
}

tr.subheader {
    background: lightgray !important;
    color: black;
}

tr.headerLastRow {
    border-bottom: 2px solid black;
}

th.rowNumber, td.rowNumber {
    text-align: right;
}

</style>
<body>
<table>

<tr class = header>
<th style = "text-align: right; ">parameters</th>
<th style = "text-align: right; ">mean</th>
<th style = "text-align: right; ">std</th>
<th style = "text-align: right; ">naive_se</th>
<th style = "text-align: right; ">mcse</th>
<th style = "text-align: right; ">ess</th>
<th style = "text-align: right; ">r_hat</th>
</tr>
<tr class = "subheader headerLastRow">
<th style = "text-align: right; ">Symbol</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Float64</th>
<th style = "text-align: right; ">Any</th>
<th style = "text-align: right; ">Any</th>
</tr>
<tr>
<td style = "text-align: right; ">sigma_angle</td>
<td style = "text-align: right; ">0.02</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4098.68</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">sigma_degrees</td>
<td style = "text-align: right; ">1.02</td>
<td style = "text-align: right; ">0.01</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">4098.68</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">sigma_distance</td>
<td style = "text-align: right; ">0.08</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">3803.99</td>
<td style = "text-align: right; ">1.0</td>
</tr>
<tr>
<td style = "text-align: right; ">sigma_y</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">0.0</td>
<td style = "text-align: right; ">785.16</td>
<td style = "text-align: right; ">1.0</td>
</tr>
</table></body></html>




Calculate predictions and visualize:

````julia
post_siga = median(chn4[:sigma_angle].value)
post_sigd = median(chn4[:sigma_distance].value)

p_angle2 = prob_angle(threshold_angle2, post_siga)
p_distance2 = prob_distance(xrngnew, distance_tolerance, overshot, post_sigd)

geom_lines2 = p_angle2 .* p_distance2

# plot
scatter(
  xnew, pnew,
  legend = false,
  color = 2,
  ylab = "Probability of Success",
  xlab = "Distance from hole (ft)")
plot!(geom_lines2, color = 2)
````


![](/post/2019-11-02_golf-turing_files/golf-turing-rev1_15_1.svg)

We can see that this adjusted first principles based model is fitting the data much better now! To add to that, it also sampled faster and more consistently during my testing. This case study really shows off the power of a bayesian approach. The modeler has the ability to expand a model using domain knowledge and craft a model that makes sense and aligns with the data generating process.

## Conclusion

If you made it this far, thanks for checking out this post! I personally appreciate all of the great scientists, applied statisticians, etc. that have created and shared the other posts I referenced as well as the team developing the Turing PPL within Julia. It's really exciting to see a PPL written entirely in one language. In my opinion, it shows the strengths of the Julia language and is an example of its promise to solve the [two-language problem](https://www.youtube.com/watch?v=B9moDuSYzGo).

## References

* [Stan: a state-of-the-art platform for statistical modeling and high-performance statistical computation](https://mc-stan.org/)
* [Model building and expansion for golf putting - Gelman](https://mc-stan.org/users/documentation/case-studies/golf.html)
* [Model building and expansion for golf putting - Carroll](https://nbviewer.jupyter.org/github/pymc-devs/pymc3/blob/master/docs/source/notebooks/putting_workflow.ipynb)
* [Bayesian golf puttings, NUTS, and optimizing your sampling function with TensorFlow Probability](https://adamhaber.github.io/post/nuts/)
* [Turing.jl: A library for robust, efficient, general-purpose probabilistic programming](https://turing.ml/dev/)
* [Announcing composable multi-threaded parallelism in Julia](https://julialang.org/blog/2019/07/multithreading)
* [ODSC East 2016 | Stefan Karpinski - "Solving the Two Language Problem"](https://www.youtube.com/watch?v=B9moDuSYzGo)

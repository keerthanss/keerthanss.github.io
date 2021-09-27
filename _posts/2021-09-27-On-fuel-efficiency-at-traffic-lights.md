---
layout: post
comments: true
title:  "On fuel efficiency at traffic lights"
date:   2021-09-27
category: "technical"
---

I recently drove a long way to meet a close friend. Since I mostly used the main roads than residential shortcut or parallel routes, I had to naturally halt at a number of traffic signals along the way. Being environment conscious, I have a thumb rule that if the red light will last longer than __30s__, then I turn off the engine. However, this time around, I wondered if I can do better. It felt like a logical basis for a simple probability question.

Suppose the ignition of car engine takes \$q\$ units, while the general rate of fuel consumption for a still vehicle be \$p\$ units. It's generally considered that \$q > p\$, which is what makes this question interesting at all. Let \$T\$ be a random variable denoting the time spent at any particular traffic signal. It may safely be assumed that \$T \sim Unif(0,120)\$, i.e. \$T\$'s distribution is uniform over the interval 0s (amounting to a green light when we arrive) to 120s. If a traffic signal exists that takes longer than 120s, rest assured I will not be choosing that route. We make one of two decisions upon arriving at a traffic light - turn off the engine, or let the engine stay on. For all the mathematical rigorists out there, observe that the case of a green light is subsumed within the case of letting the engine stay on. Our decision is based on some parameter \$t\$ -

*Decision = ON if T < t else OFF*

Let the fuel savings be denoted by \$ F_t(T) \$. Then, \$F_t(T) = (pT - q)\$ if \$T \geq t \$ else \$ 0 \$. Our goal is to maximise the expected savings.

Thus,
\[
  \mathbb{E}( F_t(t)) = \int_{0}^{t} 0 \frac{1}{120} \mathrm{d}x + \int_t^{120} (px-q) \frac{1}{120} \mathrm{d}x
\]

\[
 \implies \mathbb{E}( F_t(t)) = \frac{1}{120} (\frac{px^2}{2} - qx )\rvert_t^{120}
\]

\[
 \implies \mathbb{E}( F_t(t)) = \frac{1}{120} ( 7200p - \frac{pt^2}{2} - 120q + qt )
\]

Let \$ Z(t) =  7200p - 120q + qt - \frac{pt^2}{2} \$. This is a concave function and thus a global maxima does exist. By equating \$ Z'(t) = 0\$, we get \$ t* = \frac{q}{p} \$. A simple answer for a simple question! This form is natural given the dimensions of our parameters too.

Now let me plug in real values to allow this exercise to be useful in practice. A quick online search tells me that a standing car with a 1000cc engine consumes *0.6 L/hr*, while restarting the engine consumes *0.73ml*. This gives us *p = 0.17ml/s* and *q = 0.73ml*. Thus,
\[
  t^* = \frac{q}{p} = \frac{0.73ml}{0.17ml/s} = 4.29s
\]

This is far better than my initial thumb rule! We can round it up and deliver the verdict: **_Turn off the vehicle if you are stopping for longer than 5 seconds!_**

P.S.: The physical modeling of this question would be more nuanced. Here we have assumed *p* and *q* to be static, however, they would depend on the engine temperature too among other factors. For a more geeky engineering take on answering this question, I found this article to be rather interesting - [link](http://www.iwilltry.org/b/projects/how-many-seconds-of-idling-is-equivalent-to-starting-your-engine/).

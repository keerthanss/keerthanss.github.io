---
layout: post
comments: true
title:  "The Middle-Out of Neural Networks"
date:   2019-01-28
---

<p class="intro"><span class="dropcap">D</span>eep learning owes in part its resurgence to the advances made in computing hardware. Along with large swathes of data, deep learning research boosted immensely in recent years. However, the field has been rapacious in its consumption of resources. Parallely, recent years has also seen rapid advancements in mobile phone technology and IoT. These devices stand to benefit highly from deep learning, however their memory and compute resources are too constrained for the same. Thus, there's been active research on model compression. The idea is to train on a system with high resources, then compress and deploy smaller equivalent models that can run on resource constrained devices. Knowledge distillation is an important frontier in this space.</p>

Knowledge distillation is the process of transferring knowledge from a teacher neural network to a student neural network. The student NN is typically thinner and shallower.

<figure align="center" style="justify-content: center;">
<img src="{{ '/assets/img/knowledge_distillation_idea.png' | prepend: site.baseurl }}" alt="">
<figcaption>"Distilling" the knowledge into a smaller network for better performance</figcaption>
</figure>

Note that this is not to be confused with transfer learning. Transfer learning is using the weights of a NN as an initialization strategy to learn for a different but related task than its original design. Knowledge distillation on the other hand is not a mere copy, but it involves a process where the student network learns from the teacher. Also, considering that they are most likely of different architectures, a weight transfer is not technically meaningful either.

### So what is the process of distillation?

The answer lies in the _class probabilities_. Upon training a neural net on say, a classification task, it's not only learning the class label to be tied with the input data, but also the conditional probability distribution of all classes given that data. The end "prediction" or label that's mapped to the input, is the one with the highest probability.

We humans typically tend to learn via emulation or copying. So for example, when a schoolteacher teaches her students on how to solve some elementary math problem, the students learn not by merely observing the end result, but by the _process_ adopted by the teacher to reach that end result.

This simple idea forms the basis for knowledge distillation. The teacher model is one with a large architecture and therefore capability and capacity. The _process_ the teacher adopts to solve the problem of classification is by figuring out the probability distributions. The student network tries to emulate by _learning the **same** probability distribution_. Thus, a loss function is set up which measures how closely the student's learned probability distribution resembles the teacher's.

This idea is not exactly recent though. It has been investigated several times in the past, and there have been different schools of thought - on whether to learn the eventual logits (the feature representation of the data), or learn the softmax activations (aka probability distribution). The earliest paper which introduced this came in 2006 in a paper titled "Model Compression" by [Bucila et. al., 2006](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf).      However, the real success which popularised the idea of knowledge distillation came about when [Hinton et. al., 2014](http://arxiv.org/abs/1503.02531) introduced the concept of _temperature_ in 2014.

#### Soft vs Hard activations

[Hinton et. al., 2014](http://arxiv.org/abs/1503.02531) proposed that using the probability distribution naively is not reliable because the distributions are too polarized towards the prediction. So, the rest of the labels give very little information with regards to their distribution.

To elaborate on that, say we were given an image of an eagle. Any of the state of the art models can readily identify the image as containing an eagle, with a high degree of confidence. However, the output distribution as is, cannot really help us in telling whether the eagle looks closer to a dog or a kite. The information provided by the other classes, and as a result, by the distribution as a whole, is very minimal. This knowledge was termed as "_dark knowledge_". To expose this to a student neural network, the simple suggestion was to _flatten_ the distribution and try to make it less polarized.

Flattening can be achieved via the concept of _temperature_. As we know, given an $n$-dimensional vector $z$, the output of softmax activations is
\[
f(z) = \left \( \frac{ e^{z_{i}} } { \sum_{j=1}^{n} e^{z_{j}} } \right \)_{i=1..n}
\]

We introduce a new parameter called temperature, denoted by $T$, into this equation.
\[
f(z   ; T) = \left \( \frac{ e^{ z_{i} / T } } { \sum_{j=1}^{n} e^{ z_{j} / T } } \right \)_{i=1..n}
\]

The first equation is said to provide the _hard_ activations, while the second one with temperature incorporated into it, gives us the _soft_ activation.

We can immediately claim a few things by observing the new equation.
- First, that if $T = 1$, then we get the vanilla softmax.
- Secondly, if $T > 1$, disproportionately large logits $z_k$ will no longer enjoy a similarly scaled disproportionateness in the probability distribution, i.e. their weightage is now reduced.

One might claim that very small logits will get further diminshed and there will be no net effect. To some extent this is true, since after all we are dividing by $T$ and making it even smaller. However, it's important to note that the function is exponential. So, the effects that are possible in the lower ranges of the domain are far lesser than the effects observed in the higher ranges.

To make this point clearer, consider two logits $a_1= 100$ and $a_2=1$ respectively. Say $T = 20$. Then, considering only the numerators, $ f(a_1; T) = e^{\frac{100}{20}} = e^5 \approx 148$ and $ f(a_2; T) = e^{\frac{1}{20}} = e^{0.05} \approx 1.05$. Now, consider vanilla softmax : $f(a_1) = e^{100} \approx 2.69 \times 10^{43}$ and $f(a_2) = e^1 \approx 2.72$. Clearly, $\| f(a_1; T) - f(a_1) \|$ is larger by many orders of magnitude when compared to $\| f(a_2; T) - f(a_2) \|$. Sure, both decreased, but because the weight attached to $a_1$ decreased by a _huge_ margin, the relative weight of $a_2$ actually _increased_ as a result.

And this was exactly what we wanted! We seeked to depolarise the distribution and flatten it. A more pictorial depiction is shown below.

<figure align="center" style="justify-content: center;">
<img src="{{ '/assets/img/kd-different-temperatures.png' | prepend: site.baseurl }}" alt=""  height="90%" width="90%">
<figcaption>Activations for various values of temperature. Observe how, for the non-relevant classes, the probability increases while the probability for predicted class decreases, as temperature increases. <a href="https://towardsdatascience.com/knowledge-distillation-and-the-concept-of-dark-knowledge-8b7aed8014ac">Image credits</a> </figcaption>
</figure>

#### The loss function and performance

These _soft_ activations help a student learn the relations between classes better. For good measure, the students are also trained with the ground truth as well. Thus, the loss function for the student network can be described as
\[
  L(\hat{y}) = \alpha L(y_{teacher}, \hat{y}) + \beta L(y, \hat{y} )
\]
Thus, three new hyperparameters have been introduced - $T, \alpha, \beta$. [Hinton et. al., 2014](http://arxiv.org/abs/1503.02531) suggests to make it a weighted average so that $\beta = 1 - \alpha$. Other papers set $\beta = 1$ and $\alpha < 1$.

Empirical results confirm that the student networks learn to generalise well in this manner, and thus gain significant accuracies. For example, experiments on MNIST with a 2 layer 1200 neuron teacher and 2 layer 800 neuron student showed that while the teacher made 67 test errors, the student performed with a competitive 74 test errors (out of 10000 test samples). The generalization power was demonstrated by showing that the student performed well even if it had never seen the digit 3 during training.

It has also been shown that these networks learn much faster and perform much better than networks of the same exact architecture that learn to solve the classification task from scratch without any teacher's aid. In the previous experiments, the student without the teacher's aid made 146 test errors, but only 74 with the process of distillation. In other words, the student with the tutor performs better than the self-learned student of equal capacity.

The primary achievement here is that the student network is able to achieve competitive accuracies, in some cases even besting the teacher model, for a reduced computation cost. Thus, it requires lesser compute resources, leaves behind a smaller memory footprint, and has reduced inference time, making it a perfect candidate for deployment to resource constrained devices.

### Moving beyond

#### Ensembles

[Hinton et. al., 2014](http://arxiv.org/abs/1503.02531) demonstrate how the process of knowledge distillation can be used on a large ensemble model. Any serious Kaggler knows that the winning recipe for a competition is always a large ensemble of models. Having 30 model ensembles among the winning solutions is not uncommon. The power of ensembles is indeed not to be underestimated. In the deep learning domain, ensemble models are typically founded by training each neural network model (of possibly varying architectures and hyperparameter settings) on subsets of the data so that they learn disjoint set of features. Thus, on consolidation, we have a larger feature representation which allows the ensemble to boost performance. However, while they are fun to play around with in competitions, productionalizing and encapsulating them in an industry environment is no trivial task. This is where KD can pitch in. A singular student network can learn from the large ensemble model and achieve remarkable accuracies, higher than any individual component. Deploying a single network is logistically far simpler.

#### KD a magical formula ?

One might be tempted to think that we can use KD to compress a model drastically without impact in performance. This is not true. In the paper "Do convolutional neural nets really need to be deep and convolutional?" by [Urban et. al., 2016](https://arxiv.org/abs/1603.05691), an empirical study is carried out, wherein they perform KD on several students and compare the accuracies. The students are selected such that they can be ordered on increasing depth. It was shown that the deeper the student, the better it performed. However, the increase in performance decreased with increase in depth.The importance of convolutional layers was also demonstrated, as they outperformed networks of same depth but consisting solely of fully connected layers. Hence the answer to the question posed in its title is simply - yes.

#### Where KD falls short

Knowledge distillation works great for shallower networks, but not so much for very deep networks. To counter this, [Romero et. al., 2014](https://arxiv.org/abs/1412.6550) proposed Fitnets. The idea was simple - learn the intermediary activations as well. Thus, both the teacher and the student networks are "segmented" into "groups", and each teacher group provides its activations to the corresponding student group to learn. This is done as an initialization strategy, following which,  regular knowledge distillation is carried out. Fitnets also advocated for thinner and _deeper_ student networks. They achieved remarkable results - while a teacher with 9M parameters achieved 90.18% accuracy on CIFAR10, a fitnet student with 2.5M parameters made it to 91.61%!

#### KD in other perspectives

Until now, we discussed distilling the knowledge into a student neural network. Recent papers,  [Frosst et. al., 2017](https://arxiv.org/abs/1711.09784) and [Liu et. al., 2018](http://arxiv.org/abs/1812.10924), demonstrated how we can apply the principles and generate student decision trees too! Interpretability is quickly becoming an active area of research in deep learning. As of now, we don't really have a sound mathematical understanding of NNs, and all our empirical success mostly employs NNs as black boxes. In domains such as medical sciences, interpretability is a huge criterion. The doctor or scientist must be able to ascertain _why_ the model made some prediction, a feature deep neural networks woefully lack. This is why these works are really interesting! A decision tree is inherently easily interpretable. They provide a means of converting a black box DNN into a decision tree via knowledge distillation! The icing on the cake was that the decision tree so obtained consistently performed better than decision trees trained from scratch.

[Furnlanello et. al., 2018](https://arxiv.org/abs/1805.04770) explored the question - while knowledge distllation is meant for compression, what happens if we run it on a student that's architecturally exactly alike that of the teacher? The surprising result was that the student outperformed the teacher by a significant margin. On CIFAR100, Densenet-112-33 achieved 18.25% test error, whereas a "Born Again Neural network"-Densenet-112-33 achieved 16.95% error.

### Concluding remarks

With [Hinton et. al., 2014](http://arxiv.org/abs/1503.02531).'s seminal paper in 2014, research in knowledge distillation has become very popular, with there being many avenues and possibilities explored. A wonderful resource to explore the same is - [Awesome Knowledge Distillation](https://github.com/dkozlov/awesome-knowledge-distillation).

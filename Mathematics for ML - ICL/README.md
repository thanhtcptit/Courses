# Syllabus

## Course 1: Mathematics for Machine Learning: Linear Algebra

In this course on Linear Algebra we look at what linear algebra is and how it relates to vectors and matrices. Then we look through what vectors and matrices are and how to work with them, including the knotty problem of eigenvalues and eigenvectors, and how to use these to solve problems. Finally  we look at how to use these to do fun things with datasets - like how to rotate images of faces and how to extract eigenvectors to look at how the Pagerank algorithm works.

**Week 1: Introduction to Linear Algebra and to Mathematics for Machine Learning**

In this first module we look at how linear algebra is relevant to machine learning and data science. Then we'll wind up the module with an initial introduction to vectors. Throughout, we're focussing on developing your mathematical intuition, not of crunching through algebra or doing long pen-and-paper examples. For many of these operations, there are callable functions in Python that can do the adding up - the point is to appreciate what they do and how they work so that, when things go wrong or there are special cases, you can understand why and what to do.

**Week 2: Vectors are objects that move around space**

In this module, we look at operations we can do with vectors - finding the modulus (size), angle between vectors (dot or inner product) and projections of one vector onto another. We can then examine how the entries describing a vector will depend on what vectors we use to define the axes - the basis. That will then let us determine whether a proposed set of basis vectors are what's called 'linearly independent.' This will complete our examination of vectors, allowing us to move on to matrices in module 3 and then start to solve linear algebra problems.

**Week 3: Matrices in Linear Algebra: Objects that operate on Vectors**

Now that we've looked at vectors, we can turn to matrices. First we look at how to use matrices as tools to solve linear algebra problems, and as objects that transform vectors. Then we look at how to solve systems of linear equations using matrices, which will then take us on to look at inverse matrices and determinants, and to think about what the determinant really is, intuitively speaking. Finally, we'll look at cases of special matrices that mean that the determinant is zero or where the matrix isn't invertible - cases where algorithms that need to invert a matrix will fail.

**Week 4: Matrices make linear mappings**

In Module 4, we continue our discussion of matrices; first we think about how to code up matrix multiplication and matrix operations using the Einstein Summation Convention, which is a widely used notation in more advanced linear algebra courses. Then, we look at how matrices can transform a description of a vector from one basis (set of axes) to another. This will allow us to, for example, figure out how to apply a reflection to an image and manipulate images. We'll also look at how to construct a convenient basis vector set in order to do such transformations. Then, we'll write some code to do these transformations and apply this work computationally.

**Week 5: Eigenvalues and Eigenvectors: Application to Data Problems**

Eigenvectors are particular vectors that are unrotated by a transformation matrix, and eigenvalues are the amount by which the eigenvectors are stretched. These special 'eigen-things' are very useful in linear algebra and will let us examine Google's famous PageRank algorithm for presenting web search results. Then we'll apply this in code, which will wrap up the course.


## Course 2: Mathematics for Machine Learning: Multivariate Calculus

This course offers a brief introduction to the multivariate calculus required to build many common machine learning techniques. We start at the very beginning with a refresher on the “rise over run” formulation of a slope, before converting this to the formal definition of the gradient of a function. We then start to build up a set of tools for making calculus easier and faster. Next, we learn how to calculate vectors that point up hill on multidimensional surfaces and even put this into action using an interactive game. We take a look at how we can use calculus to build approximations to functions, as well as helping us to quantify how accurate we should expect those approximations to be. We also spend some time talking about where calculus comes up in the training of neural networks, before finally showing you how it is applied in linear regression models. This course is intended to offer an intuitive understanding of calculus, as well as the language necessary to look concepts up yourselves when you get stuck. Hopefully, without going into too much detail, you’ll still come away with the confidence to dive into some more focused machine learning courses in future.

**Week 1: What is calculus?**

Understanding calculus is central to understanding machine learning! You can think of calculus as simply a set of tools for analysing the relationship between functions and their inputs. Often, in machine learning, we are trying to find the inputs which enable a function to best match the data. We start this module from the basics, by recalling what a function is and where we might encounter one. Following this, we talk about the how, when sketching a function on a graph, the slope describes the rate of change of the output with respect to an input. Using this visual intuition we next derive a robust mathematical definition of a derivative, which we then use to differentiate some interesting functions. Finally, by studying a few examples, we develop four handy time saving rules that enable us to speed up differentiation for many common scenarios.

**Week 2: Multivariate calculus**

Building on the foundations of the previous module, we now generalise our calculus tools to handle multivariable systems. This means we can take a function with multiple inputs and determine the influence of each of them separately. It would not be unusual for a machine learning method to require the analysis of a function with thousands of inputs, so we will also introduce the linear algebra structures necessary for storing the results of our multivariate calculus analysis in an orderly fashion.

**Week 3: Multivariate chain rule and its applications**

Having seen that multivariate calculus is really no more complicated than the univariate case, we now focus on applications of the chain rule. Neural networks are one of the most popular and successful conceptual structures in machine learning. They are build up from a connected web of neurons and inspired by the structure of biological brains. The behaviour of each neuron is influenced by a set of control parameters, each of which needs to be optimised to best fit the data. The multivariate chain rule can be used to calculate the influence of each parameter of the networks, allow them to be updated during training.

**Week 4: Taylor series and linearisation**

The Taylor series is a method for re-expressing functions as polynomial series. This approach is the rational behind the use of simple linear approximations to complicated functions. In this module, we will derive the formal expression for the univariate Taylor series and discuss some important consequences of this result relevant to machine learning. Finally, we will discuss the multivariate case and see how the Jacobian and the Hessian come in to play.

**Week 5: Intro to optimisation**

If we want to find the minimum and maximum points of a function then we can use multivariate calculus to do this, say to optimise the parameters (the space) of a function to fit some data. First we’ll do this in one dimension and use the gradient to give us estimates of where the zero points of that function are, and then iterate in the Newton-Raphson method. Then we’ll extend the idea to multiple dimensions by finding the gradient vector, Grad, which is the vector of the Jacobian. This will then let us find our way to the minima and maxima in what is called the gradient descent method. We’ll then take a moment to use Grad to find the minima and maxima along a constraint in the space, which is the Lagrange multipliers method.

**Week 6: Regression**

In order to optimise the fitting parameters of a fitting function to the best fit for some data, we need a way to define how good our fit is. This goodness of fit is called chi-squared, which we’ll first apply to fitting a straight line - linear regression. Then we’ll look at how to optimise our fitting function using chi-squared in the general case using the gradient descent method. Finally, we’ll look at how to do this easily in Python in just a few lines of code, which will wrap up the course.


## Course 3: Mathematics for Machine Learning: PCA

This intermediate-level course introduces the mathematical foundations to derive Principal Component Analysis (PCA), a fundamental dimensionality reduction technique. We'll cover some basic statistics of data sets, such as mean values and variances, we'll compute distances and angles between vectors using inner products and derive orthogonal projections of data onto lower-dimensional subspaces. Using all these tools, we'll then derive PCA as a method that minimizes the average squared reconstruction error between data points and their reconstruction.

**Week 1: Statistics of Datasets**

Principal Component Analysis (PCA) is one of the most important dimensionality reduction algorithms in machine learning. In this course, we lay the mathematical foundations to derive and understand PCA from a geometric point of view. In this module, we learn how to summarize datasets (e.g., images) using basic statistics, such as the mean and the variance. We also look at properties of the mean and the variance when we shift or scale the original data set. We will provide mathematical intuition as well as the skills to derive the results. We will also implement our results in code (jupyter notebooks), which will allow us to practice our mathematical understand to compute averages of image data sets. Therefore, some python/numpy background will be necessary to get through this course. Note: If you have taken the other two courses of this specialization, this one will be harder (mostly because of the programming assignments). However, if you make it through the first week of this course, you will make it through the full course with high probability.

**Week 2: Inner Products**

Data can be interpreted as vectors. Vectors allow us to talk about geometric concepts, such as lengths, distances and angles to characterize similarity between vectors. This will become important later in the course when we discuss PCA. In this module, we will introduce and practice the concept of an inner product. Inner products allow us to talk about geometric concepts in vector spaces. More specifically, we will start with the dot product (which we may still know from school) as a special case of an inner product, and then move toward a more general concept of an inner product, which play an integral part in some areas of machine learning, such as kernel machines (this includes support vector machines and Gaussian processes). We have a lot of exercises in this module to practice and understand the concept of inner products.

**Week 3: Orthogonal Projections**

In this module, we will look at orthogonal projections of vectors, which live in a high-dimensional vector space, onto lower-dimensional subspaces. This will play an important role in the next module when we derive PCA. We will start off with a geometric motivation of what an orthogonal projection is and work our way through the corresponding derivation. We will end up with a single equation that allows us to project any vector onto a lower-dimensional subspace. However, we will also understand how this equation came about. As in the other modules, we will have both pen-and-paper practice and a small programming example with a jupyter notebook.

**Week 4: Principal Component Analysis**

We can think of dimensionality reduction as a way of compressing data with some loss, similar to jpg or mp3. Principal Component Analysis (PCA) is one of the most fundamental dimensionality reduction techniques that are used in machine learning. In this module, we use the results from the first three modules of this course and derive PCA from a geometric point of view. Within this course, this module is the most challenging one, and we will go through an explicit derivation of PCA plus some coding exercises that will make us a proficient user of PCA.


# Certificate

![Certificate](https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~C2M334BCFSRS/CERTIFICATE_LANDING_PAGE~C2M334BCFSRS.jpeg)
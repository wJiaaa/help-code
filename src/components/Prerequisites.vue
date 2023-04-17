<template>
  <div>
    <h3>Prerequisites</h3>
    <p>
      Machine Learning Crash Course does not presume or require any prior knowledge in machine
      learning. However, to understand the concepts presented and complete the exercises, we
      recommend that students meet the following prerequisites:
    </p>
    <ul>
      <li>
        You must be comfortable with variables, linear equations, graphs of functions, histograms,
        and statistical means
      </li>
      <li>
        You should be a good programmer. Ideally, you should have some experience programming in
        Python because the programming exercises are in Python. However, experienced programmers
        without <a href="https://www.python.org/">Python</a> experience can usually complete the
        programming exercises anyway.
      </li>
    </ul>
    <p>The following sections provide links to additional background material that is helpful.</p>
    <h3>Algebra</h3>
    <ul>
      <li>
        <a
          href="https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:foundation-algebra/x2f8bb11595b61c86:intro-variables/v/what-is-a-variable"
        >
          variables </a
        >,
        <a
          href="https://www.khanacademy.org/math/cc-sixth-grade-math/cc-6th-equivalent-exp/cc-6th-parts-of-expressions/v/expression-terms-factors-and-coefficients"
          >coefficients</a
        >, and
        <a href="https://www.khanacademy.org/math/algebra-home/alg-functions">functions</a>
      </li>
      <li>
        <a href="https://wikipedia.org/wiki/Linear_equation">linear equations</a>
        such as y=b+w1x1+w2x2
      </li>
      <li>
        <a href="ttps://wikipedia.org/wiki/Logarithm">logarithms</a>, and logarithmic equations such
        as y = ln(1+e^z)
      </li>
      <li>
        <a href="https://wikipedia.org/wiki/Sigmoid_function">sigmoid function</a>
      </li>
    </ul>
    <h3>Linear algebra</h3>
    <ul>
      <li>
        <a href="https://www.tensorflow.org/guide/tensor">tensor and tensor rank</a>
      </li>
      <li>
        <a href="https://wikipedia.org/wiki/Matrix_multiplication">matrix multiplication</a>
      </li>
    </ul>
    <h3>Trigonometry</h3>
    <ul>
      <li>
        (discussed as an activation function; no prior knowledge needed)
        <a href="https://reference.wolfram.com/language/ref/Tanh.html">tanh</a>
        (discussed as an
        <a href="https://developers.google.com/machine-learning/glossary#activation_function"
          >activation function</a
        >
        ; no prior knowledge needed)
      </li>
    </ul>
    <h3>Statistics</h3>
    <ul>
      <li>
        <a
          href="https://www.khanacademy.org/math/probability/data-distributions-a1/summarizing-center-distributions/v/mean-median-and-mode"
          >mean, median, outliers</a
        >, and
        <a href="https://wikipedia.org/wiki/Standard_deviation">standard deviation</a>
      </li>
      <li>ability to read a <a href="https://wikipedia.org/wiki/Histogram">histogram</a></li>
    </ul>
    <h3>Calculus (optional, for advanced topics)</h3>
    <ul>
      <li>
        concept of a <a href="https://wikipedia.org/wiki/Derivative">derivative</a> (you won't have
        to actually calculate derivatives)
      </li>
      <li>
        <a
          href="https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient"
          >gradient</a
        >
        or slope
      </li>
      <li>
        <a href="https://wikipedia.org/wiki/Partial_derivative">partial derivatives</a> (which are
        closely related to gradients)
      </li>
      <li>
        <a href="https://wikipedia.org/wiki/Chain_rule">chain rule</a> (for a full understanding of
        the
        <a href="https://developers.google.com/machine-learning/crash-course/backprop-scroll/"
          >backpropagation algorithm</a
        >
        for training neural networks)
      </li>
    </ul>
    <h3>Python Programming</h3>
    <p>
      The following Python basics are covered in
      <a href="https://docs.python.org/3/tutorial/">The Python Tutorial</a>:
    </p>
    <ul>
      <li>
        <p>
          <a href="https://docs.python.org/3/tutorial/controlflow.html#defining-functions"
            >defining and calling functions</a
          >, using positional and
          <a href="https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments"
            >keyword</a
          >
          parameters
        </p>
      </li>
      <li>
        <p>
          <a href="https://docs.python.org/3/tutorial/datastructures.html#dictionaries"
            >dictionaries</a
          >, <a href="https://docs.python.org/3/tutorial/introduction.html#lists">lists</a>,
          <a href="https://docs.python.org/3/tutorial/datastructures.html#sets">sets</a> (creating,
          accessing, and iterating)
        </p>
      </li>
      <li>
        <p>
          <a href="https://docs.python.org/3/tutorial/controlflow.html#for-statements"
            ><code translate="no" dir="ltr">for</code> loops</a
          >, <code translate="no" dir="ltr">for</code> loops with multiple iterator variables (e.g.,
          <code translate="no" dir="ltr">for a, b in [(1,2), (3,4)]</code>)
        </p>
      </li>
      <li>
        <p>
          <a href="https://docs.python.org/3/tutorial/controlflow.html#if-statements"
            ><code translate="no" dir="ltr">if/else</code> conditional blocks</a
          >
          and
          <a href="https://docs.python.org/2.5/whatsnew/pep-308.html">conditional expressions</a>
        </p>
      </li>
      <li>
        <p>
          <a href="https://docs.python.org/3/tutorial/inputoutput.html#old-string-formatting"
            >string formatting</a
          >
          (e.g., <code translate="no" dir="ltr">'%.2f' % 3.14</code>)
        </p>
      </li>
      <li>
        <p>
          variables, assignment,
          <a
            href="https://docs.python.org/3/tutorial/introduction.html#using-python-as-a-calculator"
            >basic data types</a
          >
          (<code translate="no" dir="ltr">int</code>, <code translate="no" dir="ltr">float</code>,
          <code translate="no" dir="ltr">bool</code>, <code translate="no" dir="ltr">str</code>)
        </p>
      </li>
    </ul>
    <p>A few of the programming exercises use the following more advanced Python concept:</p>
    <ul>
      <li>
        <a href="https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions"
          >list comprehensions</a
        >
      </li>
    </ul>
    <h3>Bash Terminal / Cloud Console</h3>
    <p>
      To run the programming exercises on your local machine or in a cloud console, you should be
      comfortable working on the command line:
    </p>
    <ul>
      <li>
        <a href="https://tiswww.case.edu/php/chet/bash/bashref.html">Bash Reference Manual</a>
      </li>
      <li>
        <a href="https://github.com/LeCoupa/awesome-cheatsheets/blob/master/languages/bash.sh"
          >Bash Cheatsheet</a
        >
      </li>
      <li><a href="http://www.learnshell.org/">Learn Shell</a></li>
    </ul>
  </div>
</template>

<style scoped lang="less">
p {
  margin: 16px 0;
  padding: 0;
  font: 400 16px/24px Roboto, Noto Sans, Noto Sans JP, Noto Sans KR, Noto Naskh Arabic,
    Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
}
h3 {
  margin: 32px 0 16px;
  font: 400 24px/32px Google Sans, Noto Sans, Noto Sans JP, Noto Sans KR, Noto Naskh Arabic,
    Noto Sans Thai, Noto Sans Hebrew, Noto Sans Bengali, sans-serif;
  letter-spacing: 0;
}
ul {
  margin: 0;
  padding-left: 40px;
  list-style: disc outside;
  li {
    margin: 12px 0;
    padding: 0;
    a {
      color: #1a73e8;
      cursor: pointer;
    }
  }
}
</style>

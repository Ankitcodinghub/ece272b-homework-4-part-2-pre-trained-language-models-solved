# ece272b-homework-4-part-2-pre-trained-language-models-solved
**TO GET THIS SOLUTION VISIT:** [ECE272B Homework 4 part 2-Pre-trained Language Models Solved](https://www.ankitcodinghub.com/product/ece272b-homework-4-part-2-pre-trained-language-models-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100184&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE272B Homework 4 part 2-Pre-trained Language Models Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
&nbsp;

Part 2: Pre-trained Language Models

Introduction

In the last several lectures, we have seen how convolutional neural networks (CNN) can be applied to various image-based machine learning tasks. If we view the CNN as the fundamental building block for image-related applications, one question naturally comes into our mind-what is the fun- damental building block for solving text-based problems?

This homework will also be done through Google Colab. It is split into 2 major parts (separate submissions).

Part 2

Part 2 will explore some more modern advances in the realm of text (or speech) processing. We will look at the transformer architecture, and take advantage of pre-trained networks to complete the same task that we trained from scratch in part 1. With this new approach, we also intro- duce the HuggingFace libraries Transformers and Datasets, which are regularly used platforms for transformer-based NLP today. They are built on PyTorch, so you will have to adapt away from Tensorflow a little bit.

Data Set

We will use Keras utility function: get_file to download the dataset from this URL and cache it on the file system allocated by the Colab session.

The dataset contains 16000 programming questions from Stack Overflow. Each question (E.g., “How do I sort a dictionary by value?”) is labeled with exactly one tag (Python, CSharp, JavaScript, or Java).

Here’s the code snippet for loading the dataset from URL:

<pre>from tensorflow.keras import utils
</pre>
<pre>data_url = \
    'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
</pre>
<pre>dataset = utils.get_file(
    'stack_overflow_16k.tar.gz',
    data_url,
    untar=True,
</pre>
cache_dir=”,

cache_subdir=’/PATH/TO/DIRECTORY/ON/COLAB/FILESYSTEM’) # Specify download directory

Finally, since we are using HuggingFace’s interface, we will need to change to a DataSet object. This will be provided for you, since the exact method is confusing and out of scope. In english: huggingface uses lists for text input so we need to convert to a list. Tensorflow also loads the dataset as binary strings, so we need to decode them (built in for str in python).

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
<pre>    def tfds_to_hugds(tfds, split):
        x, y = [], np.array([], dtype=int)
        for batchx, batchy in tfds:
</pre>
<pre>            x.extend([t.numpy().decode() for t in batchx])
</pre>
<pre>            y = np.append(y, batchy.numpy())
        return datasets.Dataset.from_dict({'text': x, 'labels': y}, split=split)
</pre>
<pre>    huggingface_data = datasets.DatasetDict(
        train=tfds_to_hugds(data_train, 'train'),
        validation=tfds_to_hugds(data_val, 'validation'),
        test=tfds_to_hugds(data_test, 'test')
</pre>
)

Part 2: Transformers and BERT

All Students (157B/272B)

1. (3 pts) Prepare data

<ol>
<li>(a) &nbsp;(1 pt) Download dataset. This is the same code as last week.</li>
<li>(b) &nbsp;(1 pt) Split into train, validation, and test. This is the same code as last week.</li>
<li>(c) &nbsp;(1 pt) Convert to datasets.DatasetDict format using the given conversion function (see above).</li>
</ol>
2. (5 pts) Vectorize the text. Last week we wrote a standardization function and a TextVector- ization layer from Keras. This week, we will use a Tokenizer, which was used by BERT.

<ol>
<li>(a) &nbsp;(1 pts) Use AutoTokenizer from transformers to load a BERT tokenizer. We will use distilbert-base-uncased.</li>
<li>(b) &nbsp;(2 pts) Take a training sample text, and pass it through the Tokenizer. Compare it to passing it through your TextVectorization layer. (Quickly load/adapt a TextVectoriza- tion layer, this code is the same as last week.) Describe the differences in the output. You will have to explore a bit here using built-ins like type() or .keys().</li>
<li>(c) &nbsp;(2pts)Calltokenizer.decode()andtokenizer.convert_ids_to_tokens()oninput_ids from the tokenizer output above. What do you notice? In the “convert” output, what does “##” seem to mean?

You may have to do output.input_ids[0] to unbatch it so that these functions can handle the input ids.</li>
<li>(d) &nbsp;(2 pts) Generate tokenized datasets from the text dataset by defining a function and using DataSet.map(). The keys here are to include truncation in case sentences are too long, and padding so that all inputs are the same length.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
3. (12 pts) Conceptual break: Read this blog on transformers and self-attention. You may also read the original paper, linked in the Grad section below, if that is more helpful.

<ol>
<li>(a) &nbsp;(2 pts) Explain 2 disadvantages of RNNs (LSTM) in NLP tasks. You will find this information from lecture/sources outside the blog.</li>
<li>(b) &nbsp;Reprinted for convenience is the self-attention mechanism. The letters correspond to “Query”, “Key” and “Value” matrices. dk is the hidden dimension size, but is not super important for us to look at right now.
A = softmax( d

<ul>
<li>(2 pts) Explain, in your own words , the purpose of Query/Key/Value vectors in self attention.</li>
<li>(2 pts) Explain how self-attention (eqn 1) fixes the 2 main disadvantages of LSTM.</li>
</ul>
</li>
<li>(c) &nbsp;A LSTM’s structure inherently means that current cell will only have information about past words (from the hidden state passed along). This means we get both (1) information in a causal manner, and (2) positional information by the order in which the words are processed.
<ul>
<li>(2 pts) Explain how transformers add position information to the input words. See Beast #3 in the blog.</li>
<li>(3 pts) Explain how transformers make sure the decoder stack acts in a causal manner as well (We don’t want it to “look into the future” at words that haven’t been generated yet). See Beast #4 in the blog.</li>
<li>(1 pts) BERT models are actually just the encoder portion of transformers. What do you expect their attention mask to be?</li>
</ul>
</li>
</ol>
<ol start="4">
<li>(6 pts) Load the BERT model and prepare for training
<ol>
<li>(a) &nbsp;(2pts)Loaddistilbert-base-uncasedusingAutoModelForSequenceClassification. Make sure to indicate how many classes!</li>
<li>(b) &nbsp;(3 pts) Explain the difference between AutoModelForSequenceClassification and AutoModel in view of “foundation models”.</li>
<li>(c) &nbsp;(1 pts) Set TrainingArguments. (Some) Initial arguments are given for you, which should be adequate, just set your epochs and batch size. Be sure to adjust the epochs once you start training and see the expected time!</li>
<li>(d) &nbsp;(2 pts) Load a Hugginface Trainer to use the tokenized train and validation data.</li>
</ol>
</li>
<li>(5 pts) Train (“fine-tune”) the BERT model!
<ol>
<li>(a) &nbsp;(1 pts) Train it – trainer.train(). That’s it!</li>
<li>(b) &nbsp;(2 pts) Comment on the training time compared to the LSTM. Any thoughts on why?</li>
<li>(c) &nbsp;(2 pts) HuggingFace’s interface makes the training a one line call with no arguments instead of TF’s (define loss) –&gt; (compile) –&gt; (fit). Where in our process would we have set all the hyper-parameters?</li>
</ol>
</li>
<li>(6 pts) Evaluate the performance.
<ol>
<li>(a) &nbsp;(2 pts) Report the test set accuracy. You can get the predicted logits from trainer.predict(…). How does this compare to your best LSTM accuracy?</li>
<li>(b) &nbsp;(2 pts) Take the following questions as input, what are the predicted tags? i. “how do I extract keys from a dict into a list?”
ii. “debug public static void main(string[] args) …”

Do you agree? A function is provided to correctly get the logits for you, since after training some of the the model will be on the gpu, but the text you make will be on the cpu.
</li>
<li>(c) &nbsp;(2 pts) What is the most annoying aspect about using/training/evaluating BERT? For our problem, is this annoyance worth the performance?</li>
</ol>
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
QKT

</div>
<div class="column">
)V (1)

</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
<div class="layoutArea">
<div class="column">
k

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Grad/Extra Credit (272B)

1. (11 Points) Understanding the how and why of Transformers.

<ol>
<li>(a) &nbsp;Read the original transformers paper.</li>
<li>(b) &nbsp;(3 pts) Explain the main computational limitation of transformers/BERT, especially in
comparison with LSTM. Be thorough.

√
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
(c) (2 pts) Explain the purpose of

</div>
<div class="column">
dk term in equation 1.

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol start="4">
<li>(d) &nbsp;(4 pts) Instead of pure self-attention, the authors used multi-head-attention. Ex- plain the purpose of this mechanism i) intuitively, and ii) in terms of computational complexity.</li>
<li>(e) &nbsp;(2 pts) It’s not the easiest to see from the paper, but their diagram shows a “add and norm” layer within each transformer layer. Explain the “add” component to the best of your understanding. (Hint – is something here similar to Recurrent networks? Residual networks?). If you are unsure, that is OK. Try to find some intuition and defend your stance.</li>
</ol>
</div>
</div>
</div>

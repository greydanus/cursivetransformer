# Generating Cursive with a Transformer

Sam Greydanus

## Making a dataset

Let's construct a dataset of cursive pen strokes for training a handwriting model. I don't have an e-pen or any special hardware. Also, someday I want to allow people to clone their own handwriting in a demo. Thus this is a strictly mouse-based interface.

The `collect.html` is a simple webpage that allows users to upload examples of cursive handwriting, position those images in the tracing region, trace them with a pen, annotate the author and ASCII characters, and export the result as a JSON file. This interface was build with the help of Claude Sonnet 3.5.

* _Made it to page 36 of the [Zaner-Bloser cursive practice workbook](static/Zaner-Bloser.pdf)

![collect](static/collect.png)


## Preprocessing and Tokenization

Our raw dataset consists of a large JSON file consisting of examples. Each example contains ASCII characters, stroke information, and some metadata like who the author was. Let's visualize one of these examples: this particular one was traced from a screenshot taken of the [Zaner-Bloser cursive practice workbook](static/Zaner-Bloser.pdf). Since we have to represent (`dx`, `dy`, `magnitude`, and `is_pen_down`) for every step, we opt to unroll each step into three tokens: the first represents `dx`, the second represents `dy`, and the third represents a combination of `magnitude` and `is_pen_down`. This is a little messy, but it allows us to leave the boilerplate Transformer training code completely unchanged (there's an alternative that involves using a custom embedding but it's more trouble than it's worth).

![tokenizer](static/encode_decode.png)

## Training and Logging

The model definition is Karpathy's `makemore` Transformer architecture plus cross-attention for including ASCII data. The training infrastructure is also based upon the `makemore` repo. However, we've added Weights and Biases logging (screenshot below). This logging also uploads samples for qualitative evaluation.

![tokenizer](static/wandb.png)
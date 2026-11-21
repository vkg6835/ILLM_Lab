# Assignment 9 - Pretrained Embeddings for English-to-Hindi MT

Vishal Gautam, U23AI127

The exact Lab 8 splits were reused and hash-verified: 128,605 training,
16,075 validation and 16,075 test pairs. Bins use original English word
length: 1-10, 11-19, 20-29 and 30+. Exactly 20 belongs to bin 3.

Both models: OpenNMT-py 3.5.1, 2 encoder/2 decoder Transformer layers,
hidden size 128, 4 heads, FFN 512, 8000 updates, label smoothing 0.1,
Adam with Noam schedule, dropout 0.1; pretrained embeddings fine-tuned.
Noam factor 0.2, 500 warmup updates, gradient norm clipping at 1.
Validation runs every 1000 updates. Effective peak learning rate is about 0.00079.
Validation perplexity selects the checkpoint; beam size 5 translates test.

FastText uses official 300D Wiki word vectors and a capped 12,000-word
vocabulary per language. English scan limit: 250000 rows (0 means full).
Hindi vectors were fully scanned. Uncovered rows use seeded random vectors.
BPEmb uses official English/Hindi 10,000-piece segmenters and 300D vectors.
Text is encoded with BPEmb before vector lookup. Training-type PCA reduces
both variants to 128D, followed by equal row scaling. PCA is saved.

## Actual test results

| Embeddings | Bin | Test pairs | BLEU | Brevity penalty | Length ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| fasttext | bin1 | 3232 | 8.8565 | 1.0000 | 1.0888 |
| fasttext | bin2 | 6729 | 7.0157 | 1.0000 | 1.1790 |
| fasttext | bin3 | 4165 | 5.7543 | 1.0000 | 1.2607 |
| fasttext | bin4 | 1949 | 5.3525 | 1.0000 | 1.3764 |
| fasttext | overall | 16075 | 6.2626 | 1.0000 | 1.2455 |
| bpemb | bin1 | 3232 | 8.3931 | 0.8779 | 0.8848 |
| bpemb | bin2 | 6729 | 6.4571 | 0.9105 | 0.9143 |
| bpemb | bin3 | 4165 | 5.8678 | 0.8977 | 0.9026 |
| bpemb | bin4 | 1949 | 5.9126 | 0.9413 | 0.9430 |
| bpemb | overall | 16075 | 6.2756 | 0.9110 | 0.9147 |

## Bin-wise observations

bin1: BPEmb-minus-FastText BLEU is -0.4634. FastText length ratio is 1.0888; BPEmb length ratio is 0.8848. Ratios below one indicate shorter output than the reference and may help explain the brevity penalty, but do not alone identify an error source.

bin2: BPEmb-minus-FastText BLEU is -0.5585. FastText length ratio is 1.1790; BPEmb length ratio is 0.9143. Ratios below one indicate shorter output than the reference and may help explain the brevity penalty, but do not alone identify an error source.

bin3: BPEmb-minus-FastText BLEU is +0.1135. FastText length ratio is 1.2607; BPEmb length ratio is 0.9026. Ratios below one indicate shorter output than the reference and may help explain the brevity penalty, but do not alone identify an error source.

bin4: BPEmb-minus-FastText BLEU is +0.5601. FastText length ratio is 1.3764; BPEmb length ratio is 0.9430. Ratios below one indicate shorter output than the reference and may help explain the brevity penalty, but do not alone identify an error source.

fasttext: strongest bin is bin1 (8.8565); weakest is bin4 (5.3525).

bpemb: strongest bin is bin1 (8.3931); weakest is bin3 (5.8678).

## Coverage and limitations

| variant   | side   |   vocab_size |   pretrained_types |   training_token_vocabulary_coverage |   training_token_pretrained_coverage |   training_unknown_token_rate | aligned_shape   |
|:----------|:-------|-------------:|-------------------:|-------------------------------------:|-------------------------------------:|------------------------------:|:----------------|
| fasttext  | src    |        12008 |               8797 |                             0.932937 |                             0.832781 |                    0.067063   | [12008, 128]    |
| fasttext  | tgt    |        12008 |              10910 |                             0.927449 |                             0.896491 |                    0.0725512  | [12008, 128]    |
| bpemb     | src    |         9624 |               9614 |                             1        |                             1        |                    0.00244745 | [9624, 128]     |
| bpemb     | tgt    |         9544 |               9535 |                             1        |                             1        |                    0.00386722 | [9544, 128]     |

Scores measure these two complete pipelines. Tokenization, vocabulary size,
PCA loss and partial English vector coverage confound an embedding-only claim.
BPEmb additionally lowercases inputs and replaces digits with zero; FastText
preserves them. BPEmb encoding uses canonical IDs so unknown surface forms
map to `<unk>` consistently. Original Hindi references are never normalized.
Pretrained coverage includes the supplied BPEmb `<unk>` vector; the separately
reported unknown-token rate measures tokens whose surface information is lost.
The fixed update budget may not traverse all training rows. Low scores describe
a small educational baseline, not a deployable translation system. Lab 8 used
5,000 updates and is not an equal-budget comparator. Test was not used for
checkpoint selection. Overall BLEU is a separate corpus calculation.

Perplexity uses different word/subword prediction units, so it selects checkpoints
within each variant. BLEU provides the comparison across the translators.

## Actual translation examples

### fasttext, bin1

English: A Frobenius norm how many of you get that now?

Hindi reference: एक फ्रोबेनियस मानदंड है कि आप में से कितनो को अब वह मिलता है?

Prediction: यह एक &lt;unk&gt; है कि आप वास्तव में क्या करते हैं ?

### fasttext, bin2

English: Colourful birds capture the heart of tourists by their sweet songs .

Hindi reference: रंग - बिरंगे पक्षी अपने मधुर गीत - संगीत से भ्रमणकारियों का मन मोह लेते हैं ।

Prediction: &lt;unk&gt; &lt;unk&gt; के पर्यटकों को अपनी &lt;unk&gt; से &lt;unk&gt; ।

### fasttext, bin3

English: National Institution of Nutrition , Hyderabad has researched that the seeds of fenugreek are panacea for the diabetes patients .

Hindi reference: नेशनल इंस्टीट्‍यूट ऑफ न्यूटीशन हैदराबाद के वैज्ञानिकों का शोध है कि मेंथी के दाने मधुमेह के रोगियों के लिए रामबाण हैं

Prediction: &lt;unk&gt; राष्‍ट्रीय उद्यान के &lt;unk&gt; ने बताया है कि इन रोगियों के लिए इन रोगियों के लिए &lt;unk&gt; हैं ।

### fasttext, bin4

English: The celebrated Chinese traveler Huein Tsang visited the state in the middle of 7th century A.D. Nearly half (44%) of the state is forested and offers a unique panorama of flora and fauna.

Hindi reference: प्रसिद्ध चीनी यात्री ह्युनसांग ने 7वीं शताब्दी ई. के मध्य राज्य की यात्रा की।

Prediction: &lt;unk&gt; राज्य के मध्य प्रदेश के मध्य प्रदेश के मध्य में मनाया जाता है . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

### bpemb, bin1

English: A Frobenius norm how many of you get that now?

Hindi reference: एक फ्रोबेनियस मानदंड है कि आप में से कितनो को अब वह मिलता है?

Prediction: एक ((|प्रोग्राम|function)) क्या है?

### bpemb, bin2

English: Colourful birds capture the heart of tourists by their sweet songs .

Hindi reference: रंग - बिरंगे पक्षी अपने मधुर गीत - संगीत से भ्रमणकारियों का मन मोह लेते हैं ।

Prediction: पक्षियों की खूबसूरती की खूबसूरती के चारों ओर आते हैं ।

### bpemb, bin3

English: National Institution of Nutrition , Hyderabad has researched that the seeds of fenugreek are panacea for the diabetes patients .

Hindi reference: नेशनल इंस्टीट्‍यूट ऑफ न्यूटीशन हैदराबाद के वैज्ञानिकों का शोध है कि मेंथी के दाने मधुमेह के रोगियों के लिए रामबाण हैं

Prediction: राष् ⁇ ट्रीय पोषक तत्वों की रोकथाम की जाती है , जो मधुमेह के लिए , मधुमेह के रोगियों के लिए किया जाता है ।

### bpemb, bin4

English: The celebrated Chinese traveler Huein Tsang visited the state in the middle of 7th century A.D. Nearly half (44%) of the state is forested and offers a unique panorama of flora and fauna.

Hindi reference: प्रसिद्ध चीनी यात्री ह्युनसांग ने 7वीं शताब्दी ई. के मध्य राज्य की यात्रा की।

Prediction: 00वीं शताब्दी के मध्य प्रदेश के मध्य प्रदेश के बीच, 00वीं शताब्दी के मध्य प्रदेश के मध्य प्रदेश के लगभग 00 किलोमीटर दूर है।

## References

- Previous solution: ../Lab8/Assignment8_English_Hindi_MT.ipynb
- FastText vectors: https://fasttext.cc/docs/en/pretrained-vectors.html
- BPEmb: https://github.com/bheinzerling/bpemb
- OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
- SacreBLEU: https://github.com/mjpost/sacrebleu

Scoring signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0

import logging
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from .origin_bert.modeling_bert import BertConfig, BertPreTrainedModel
from .origin_bert.modeling_utils import load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP

logger = logging.getLogger()
StepFunctionType = Callable[
    [torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]
]

class CaptionPreTrainedModel(BertPreTrainedModel):
    """ Expand base class for image captioning modeling.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'bert'

    def __init__(self, config, *inputs, **kwargs):
        super(CaptionPreTrainedModel, self).__init__(config, *inputs, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False

    def generate(
        self,
        input_ids=None,
        max_length=None,
        do_sample=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
    ):
        r""" Generates sequences for models with a LM head. The method currently supports greedy or penalized greedy decoding, sampling with top-k or nucleus sampling
        and beam-search.
        Adapted in part from `Facebook's XLM beam search code`_.
        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529
        Parameters:
            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.
            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between 1 and infinity. Default to 20.
            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Default to greedy sampling.
            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.
            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictely positive. Default to 1.0.
            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.
            bos_token_id: (`optional`) int
                Beginning of sentence token if no prompt is provided. Default to 0.
            eos_token_ids: (`optional`) int or list of int
                End of sequence token or list of tokens to stop the generation. Default to 0.
            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.
            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.
        Examples::
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)  # do greedy decoding without beam search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, do_sample=True, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[0][i], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id, num_beams=3)  # generate sequences using greedy beam search decoding (3 beams)
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences using using greedy search
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`)"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_ids = eos_token_ids if eos_token_ids is not None else self.config.eos_token_ids
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictely positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictely positive integer."
        assert temperature > 0, "`temperature` should be strictely positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, "`bos_token_id` should be a positive integer."
        assert isinstance(pad_token_id, int) and pad_token_id >= 0, "`pad_token_id` should be a positive integer."
        assert isinstance(eos_token_ids, (list, tuple)) and (
            e >= 0 for e in eos_token_ids
        ), "`eos_token_ids` should be a positive integer or a list/tuple of positive integers."
        assert length_penalty > 0, "`length_penalty` should be strictely positive."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictely positive integer."

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # current position and vocab size
        cur_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size

        if num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
            input_ids = input_ids.contiguous().view(
                batch_size * num_return_sequences, cur_len
            )  # (batch_size * num_return_sequences, cur_len)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len,
                max_length,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        if num_return_sequences != 1:
            for i in range(len(output)):
                output[i] = output[i].view(batch_size, num_return_sequences, -1)
        return output

    def _decode_step(self, input_ids, past):
        model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        token_len = outputs[0].shape[1]
        if self.od_labels_len == 0:
            next_token_idx = token_len - 1
        else:
            if token_len == 2:
                assert self._do_output_past(outputs)
                next_token_idx = 1
            else:
                next_token_idx = token_len - self.od_labels_len - 1

        next_token_logits = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
        assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

        # if model has past, then set the past variable to speed up decoding
        if self._do_output_past(outputs):
            past = outputs[1]
        return next_token_logits, past

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = input_ids.new(batch_size).fill_(1) # [batch_size] filled with 1

        # log of scores for each sentence in the batch
        logprobs = []

        past = None

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            outputs = self(**model_inputs)
            if cur_len == 1:
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1
            assert outputs[0].shape[1] == token_len

            next_token_logits = outputs[0][:, next_token_idx, :]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            #for t in input_ids:
                #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        num_keep_best = self.num_keep_best
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        TOPN_PER_BEAM = 2

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        # panmz: past is a tuple containing all layer outputs, (batch_size * num_beams, token_len, hid_size)
        # panmz: first dim will be permute such that it corresponds to the right sentences
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past)
            # panmz: this will call forward(), the model_inputs will not include
            # panmz: param 'is_decode', so it will be False by default
            # panmz: so forward() will further call encode_forward()
            outputs = self(**model_inputs)
            # tuple, first is vocab distribution. (batch_size * num_beams, len, vocab_size)
            # second is a tuple of all layer states
            if cur_len == 1: # first iteration
                token_len = 2 + self.od_labels_len
                next_token_idx = 1
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len + 1 + self.od_labels_len
                    next_token_idx = cur_len
                else:
                    token_len = 2
                    next_token_idx = 1

            assert outputs[0].shape[1] == token_len
            scores = outputs[0][:, next_token_idx, :]  # (batch_size * num_beams, vocab_size)
            assert outputs[0].shape[1] == model_inputs['input_ids'].shape[1]

            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [TOPN_PER_BEAM] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=TOPN_PER_BEAM)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, TOPN_PER_BEAM)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, TOPN_PER_BEAM).to(next_words.device)
                next_words = next_words.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * num_beams, dim=1, largest=True, sorted=True)
            # panmz: largest TOPN_PER_BEAM * num_beams scores and indexes of all beams
            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):
                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []
                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    # idx range from 0 to num_beams * vocab_size - 1
                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past: # layer_past: (batch_size * num_beams, token_len, hid_dim)
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 1st position
                    reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            # hypotheses.hyp: list of (score, hyp). max len is num_keep_best
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # panmz: [all_best] has total batch_size [best], each [best] contains
        # panmz: num_keep_best hypotheses at most.

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]

        # panmz: decoded of size (batch_size, num_keep_best, max_length)
        # panmz: decoded[i, j, :] looks like ['[BOS]', 'w1', ..., 'wn', '[EOS]', '[PAD]', ..., '[PAD]']
        # panmz: logprobs of size (batch_size, num_keep_best)
        # panmz: logprobs[i, j] is the log probabilities of the jth best hypothese in ith batch
        return decoded, logprobs


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = [] # panmz: a priority queue stores maximum scores [(scores, hyp), ...]
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list. When reach the end of the sentence
        params:
            hyp: current sentence
            sum_logprobs: score for next token
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score: # not full or need pop
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


class ConstrainedBeamSearch(object):
    r"""
    Implements Constrained Beam Search for decoding the most likely sequences conditioned on a
    Finite State Machine with specified state transitions.
    """

    def __init__(
        self,
        eos_token_ids: List[int],
        max_steps: int = 20,
        beam_size: int = 5,
        per_node_beam_size: Optional[int] = None,
        use_hypo: bool = False,
        tokenizer=None,
    ):
        self._eos_token_ids = eos_token_ids
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or self.beam_size
        self.num_keep_best = 1
        self.length_penalty = 1
        self.use_hypo = use_hypo
        self.tokenizer = tokenizer

    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: List[torch.Tensor],
        step: StepFunctionType,
        fsm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Given a starting state, a step function, and an FSM adjacency matrix, apply Constrained
        Beam Search to find most likely target sequences satisfying specified constraints in FSM.
        .. note::
            If your step function returns ``-inf`` for some log probabilities
            (like if you're using a masked log-softmax) then some of the "best"
            sequences returned may also have ``-inf`` log probability. Specifically
            this happens when the beam size is smaller than the number of actions
            with finite log probability (non-zero probability) returned by the step function.
            Therefore if you're using a mask you may want to check the results from ``search``
            and potentially discard sequences with non-finite log probability.
        Parameters
        ----------
        start_predictions : torch.Tensor
            A tensor containing the initial predictions with shape ``(batch_size, )``. These are
            usually just ``@@BOUNDARY@@`` token indices.
        start_state : ``Dict[str, torch.Tensor]``
            The initial state passed to the ``step`` function. Each value of the state dict
            should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
            number of dimensions.
        step : ``StepFunctionType``
            A function that is responsible for computing the next most likely tokens, given the
            current state and the predictions from the last time step. The function should accept
            two arguments. The first being a tensor of shape ``(group_size,)``, representing the
            index of the predicted tokens from the last time step, and the second being the
            current state. The ``group_size`` will be ``batch_size * beam_size * num_fsm_states``
            except in the initial step, for which it will just be ``batch_size``. The function is
            expected to return a tuple, where the first element is a tensor of shape
            ``(group_size, vocab_size)`` containing the log probabilities of the tokens for the
            next step, and the second element is the updated state. The tensor in the state should
            have shape ``(group_size, *)``, where ``*`` means any other number of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, num_fsm_states, beam_size, max_steps)``
            and ``log_probabilities`` has shape ``(batch_size, num_fsm_states, beam_size)``.
        """
        # shape: (batch_size, num_fsm_states, num_fsm_states, vocab_size)
        batch_size, num_fsm_states, _, vocab_size = fsm.size()

        # generated hypotheses
        generated_hyps = [
            [BeamHypotheses(self.num_keep_best, self.max_steps, self.length_penalty, early_stopping=False)
            for _ in range(num_fsm_states)]
            for bb in range(batch_size)
        ]

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, num_fsm_states, beam_size) tensors. One for each time step. None for
        # the first. Stores the index n for the parent prediction.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop because we are going
        # from a single decoder input (the output from the encoder) to the top `beam_size`
        # decoder outputs per FSM state. On the other hand, within the main loop we are going
        # from the `beam_size` elements of the beam (per FSM state) to `beam_size`^2 candidates
        # from which we will select the top `beam_size` elements for the next iteration.

        curr_ids = (
            start_predictions.expand(batch_size, self.beam_size*num_fsm_states)
            .reshape(batch_size*self.beam_size*num_fsm_states, 1)
        )
        # shape: start_class_log_probabilities (batch_size, vocab_size)
        start_class_logits, state = step(curr_ids, start_state)
        start_class_log_probabilities = torch.nn.functional.log_softmax(start_class_logits, dim=-1)
        start_class_log_probabilities = start_class_log_probabilities[:batch_size, :]
        vocab_size = start_class_log_probabilities.size(-1)

        start_state_predictions = start_class_log_probabilities.view(
            batch_size, 1, vocab_size
        ).expand(batch_size, num_fsm_states, vocab_size)

        start_state_predictions = start_state_predictions.masked_fill(
            (1 - fsm[:, 0, :, :]).to(dtype=torch.bool), float("-inf")
        )

        # (batch_size, num_fsm_states, beam_size)
        start_top_log_probabilities, start_predicted_classes = start_state_predictions.topk(
            self.beam_size
        )
        # shape: (batch_size, num_fsm_states, beam_size)
        last_log_probabilities = start_top_log_probabilities

        predictions.append(start_predicted_classes.view(batch_size, -1))

        log_probs_after_end = torch.full((1, vocab_size), float("-inf")).to(
            start_predictions.device
        )
        log_probs_after_end[:, self._eos_token_ids] = 0.0

        #state = {
            #key: _enlarge_single_tensor(value, batch_size, num_fsm_states, self.beam_size)
            #for (key, value) in state.items()
        #}

        step_state_mask = fsm.view(
            batch_size, num_fsm_states, num_fsm_states, 1, vocab_size
        ).expand(batch_size, num_fsm_states, num_fsm_states, self.beam_size, vocab_size)

        curr_len = curr_ids.shape[1]
        for timestep in range(self.max_steps - curr_len - 1):
            # shape: (batch_size * beam_size * num_fsm_states, )
            last_predictions = predictions[-1].reshape(
                batch_size * self.beam_size * num_fsm_states
            )
            cur_finished = (last_predictions==self._eos_token_ids[0])
            for eos_token in self._eos_token_ids[1:]:
                cur_finished = (cur_finished | (last_predictions==eos_token))
            if cur_finished.all():
                break

            curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)

            class_logits, state = step(curr_ids, state)
            class_log_probabilities = torch.nn.functional.log_softmax(class_logits, dim=-1)
            #last_predictions_expanded = (
                #last_predictions.view(-1)
                #.unsqueeze(-1)
                #.expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            #)
            cur_finished_expanded = (
                cur_finished.unsqueeze(-1)
                .expand(batch_size * num_fsm_states * self.beam_size, vocab_size)
            )

            cleaned_log_probabilities = torch.where(
                #last_predictions_expanded == self._eos_token_ids,
                cur_finished_expanded,
                log_probs_after_end,
                class_log_probabilities,
            )
            cleaned_log_probabilities = cleaned_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, vocab_size
            )

            device = start_predictions.device
            restricted_predicted_classes = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_log_probs = torch.FloatTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)
            restricted_beam_indices = torch.LongTensor(
                batch_size, num_fsm_states, self.beam_size
            ).to(start_predictions.device)

            expanded_last_log_probabilities = last_log_probabilities.view(
                batch_size, num_fsm_states, self.beam_size, 1
            ).expand(batch_size, num_fsm_states, self.beam_size, self.per_node_beam_size)

            for i in range(num_fsm_states):
                # shape (batch_size, num_fsm_states, self.beam_size, vocab_size)
                state_log_probabilities = cleaned_log_probabilities

                state_log_probabilities = state_log_probabilities.masked_fill(
                    (1 - step_state_mask[:, :, i, :, :]).to(dtype=torch.bool), -1e20
                )
                top_log_probabilities, predicted_classes = state_log_probabilities.topk(
                    self.per_node_beam_size
                )
                summed_top_log_probabilities = (
                    top_log_probabilities + expanded_last_log_probabilities
                )
                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_summed = summed_top_log_probabilities.reshape(batch_size, -1)

                # shape: (batch_size, old_num_fsm_states * beam_size * per_node_beam_size)
                reshaped_predicted_classes = predicted_classes.reshape(batch_size, -1)

                if not self.use_hypo:
                    # shape (batch_size, beam_size)
                    state_beam_log_probs, state_beam_indices = reshaped_summed.topk(self.beam_size)
                    # shape (batch_size, beam_size)
                    state_predicted_classes = reshaped_predicted_classes.gather(1, state_beam_indices)
                else:
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_beam_log_probs, candidate_beam_indices = reshaped_summed.topk(
                            self.beam_size*self.per_node_beam_size, sorted=True, largest=True)
                    # shape (batch_size, beam_size*per_node_beam_size)
                    candidate_predicted_classes = reshaped_predicted_classes.gather(1, candidate_beam_indices)
                    next_batch_beam = []
                    for batch_ex in range(batch_size):
                        next_sent_beam = []
                        for word_id, beam_id, log_prob in zip(candidate_predicted_classes[batch_ex],
                                    candidate_beam_indices[batch_ex],
                                    candidate_beam_log_probs[batch_ex]):
                            if word_id.item() in self._eos_token_ids:
                                generated_hyps[batch_ex][i].add(
                                    curr_ids[batch_ex * self.beam_size*num_fsm_states + beam_id/self.per_node_beam_size, :].clone(),
                                    log_prob.item()
                                )
                            else:
                                next_sent_beam.append((word_id, beam_id, log_prob))
                            if len(next_sent_beam) == self.beam_size:
                                break
                        assert len(next_sent_beam) == self.beam_size
                        next_batch_beam.extend(next_sent_beam)
                    state_predicted_classes = torch.tensor([x[0] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_indices = torch.tensor([x[1] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)
                    state_beam_log_probs = torch.tensor([x[2] for x in next_batch_beam],
                            device=device).reshape(batch_size, self.beam_size)

                restricted_predicted_classes[:, i, :] = state_predicted_classes
                restricted_beam_indices[:, i, :] = state_beam_indices
                restricted_beam_log_probs[:, i, :] = state_beam_log_probs

            restricted_predicted_classes = restricted_predicted_classes.view(batch_size, -1)
            predictions.append(restricted_predicted_classes)

            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer.view(batch_size, -1))

            last_log_probabilities = restricted_beam_log_probs.view(batch_size, num_fsm_states, -1)

            def track_back_state(state_tensor):
                _, *last_dims = state_tensor.size()
                # shape: (batch_size, beam_size, *)
                expanded_backpointer = backpointer.view(
                    batch_size, num_fsm_states * self.beam_size, *([1] * len(last_dims))
                ).expand(batch_size, num_fsm_states * self.beam_size, *last_dims)

                # shape: (batch_size * beam_size, *)
                return (
                    state_tensor.reshape(batch_size, num_fsm_states * self.beam_size, *last_dims)
                    .gather(1, expanded_backpointer)
                    .reshape(batch_size * num_fsm_states * self.beam_size, *last_dims)
                )
            # reorder states
            if state is not None:
                state = tuple(track_back_state(value) for value in state)
            curr_ids = track_back_state(curr_ids)

        last_predictions = predictions[-1].reshape(
            batch_size * self.beam_size * num_fsm_states
        )
        curr_ids = torch.cat([curr_ids, last_predictions.unsqueeze(-1)], dim=1)
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        all_predictions = all_predictions.view(batch_size, num_fsm_states, self.beam_size, -1)
        assert (all_predictions == curr_ids.reshape(batch_size, num_fsm_states,
                self.beam_size, -1)[:,:,:,1:]).all()

        if self.use_hypo:
            decoded = all_predictions.new(batch_size, num_fsm_states, 1,
                    self.max_steps).fill_(self._eos_token_ids[0])
            scores = last_log_probabilities.new(batch_size, num_fsm_states,
                    1).fill_(-1e5)
            for batch_ex in range(batch_size):
                for i in range(num_fsm_states):
                    beam = all_predictions[batch_ex, i, 0, :]
                    log_prob = last_log_probabilities[batch_ex, i, 0]
                    generated_hyps[batch_ex][i].add(
                        beam.clone(),
                        log_prob.item()
                    )
                    hyps = generated_hyps[batch_ex][i].hyp
                    assert len(hyps) == 1
                    score, sent = hyps[0]
                    decoded[batch_ex, i, 0, :len(sent)] = sent
                    scores[batch_ex, i, 0] = score
            all_predictions = decoded
            last_log_probabilities = scores

        # pad to the same length, otherwise DataParallel will give error
        pad_len = self.max_steps - all_predictions.shape[-1]
        if pad_len > 0:
            padding_ids = all_predictions.new(
                    batch_size, num_fsm_states, self.beam_size,
                    pad_len).fill_(self._eos_token_ids[0])
            all_predictions = torch.cat([all_predictions, padding_ids], dim=-1)

        return all_predictions, last_log_probabilities


def select_best_beam_with_constraints(
    beams: torch.Tensor,
    beam_log_probabilities: torch.Tensor,
    given_constraints: torch.Tensor,
    min_constraints_to_satisfy: int,
    eos_token_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Select the best beam which satisfies specified minimum constraints out of a total number of
    given constraints.
    .. note::
        The implementation of this function goes hand-in-hand with the FSM building implementation
        in :meth:`~updown.utils.constraints.FiniteStateMachineBuilder.build` - it defines which
        state satisfies which (basically, how many) constraints. If the "definition" of states
        change, then selection of beams also changes accordingly.
    Parameters
    ----------
    beams: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size, max_decoding_steps)`` containing
        decoded beams by :class:`~updown.modules.cbs.ConstrainedBeamSearch`. These beams are
        sorted according to their likelihood (descending) in ``beam_size`` dimension.
    beam_log_probabilities: torch.Tensor
        A tensor of shape ``(batch_size, num_states, beam_size)`` containing likelihood of decoded
        beams.
    given_constraints: torch.Tensor
        A tensor of shape ``(batch_size, )`` containing number of constraints given at the start
        of decoding.
    min_constraints_to_satisfy: int
        Minimum number of constraints to satisfy. This is either 2, or ``given_constraints`` if
        they are less than 2. Beams corresponding to states not satisfying at least these number
        of constraints will be dropped. Only up to 3 supported.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Decoded sequence (beam) which has highest likelihood among beams satisfying constraints.
    """
    batch_size, num_states, beam_size, max_decoding_steps = beams.size()

    best_beams: List[torch.Tensor] = []
    best_beam_log_probabilities: List[torch.Tensor] = []

    for i in range(batch_size):
        # fmt: off
        valid_states = [
            s for s in range(2 ** given_constraints[i].item())
            if bin(s).count("1") >= min(given_constraints[i], min_constraints_to_satisfy)
        ]
        # fmt: on

        valid_beams = beams[i, valid_states, 0, :]
        valid_length = torch.ones_like(valid_beams)
        for eos_token_id in eos_token_ids:
            valid_length = valid_length.mul(valid_beams.ne(eos_token_id).long())
        valid_length = valid_length.sum(1) + 1
        valid_beam_log_probabilities = beam_log_probabilities[i, valid_states, 0] / valid_length

        selected_index = torch.argmax(valid_beam_log_probabilities)
        best_beams.append(valid_beams[selected_index, :] )
        best_beam_log_probabilities.append(valid_beam_log_probabilities[selected_index])

    # shape: (batch_size, max_decoding_steps)
    return (torch.stack(best_beams).long().to(beams.device),
            torch.stack(best_beam_log_probabilities).to(beams.device))
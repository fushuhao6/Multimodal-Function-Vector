from tqdm.auto import tqdm
import torch
from .extract_utils import inference_model
from .eval_utils import compute_individual_token_rank
from .utils import get_answer_id, construct_training_labels


class FunctionVectorModel(torch.nn.Module):
    def __init__(self, model, edit_layer, function_vector, model_config, submodel='language_model'):
        super().__init__()
        self.model = model
        self.edit_layer = edit_layer
        self.model_config = model_config
        self.submodel = submodel
        self.hook_handles = []
        self.idx = None  # Temporarily set per forward call

        self.device = model.device
        if 'resid_out_dim' in model_config[submodel]:
            model_resid_out_dim = model_config[submodel]['resid_out_dim']
        else:
            model_resid_out_dim = model_config[submodel]['resid_dim']

        for param in self.model.parameters():
            param.requires_grad = False

        function_vector = function_vector.reshape(1, model_resid_out_dim).to(self.device)
        self.function_vector = torch.nn.Parameter(function_vector, requires_grad=True)

        self._register_hooks()

    def _find_target_layer(self):
        target_layer_name = self.model_config[self.submodel]['layer_hook_names'][self.edit_layer]
        for name, module in self.model.named_modules():
            if target_layer_name in name:
                # print(f"Found target layer: {name}")
                return module
        return None

    def _register_hooks(self):
        target_layer = self._find_target_layer()

        if target_layer is None:
            print(f"Warning: Could not find target layer with ID {self.edit_layer}")
            return

        original_forward = target_layer.forward

        def new_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if self.idx is None:
                raise ValueError("FunctionVectorModel.forward must be called with `idx` argument.")

            def inject(t):
                # t: (B, S, H)
                B, S, H = t.shape
                # Build a selector for the time dimension only
                sel = torch.zeros((B, S, 1), dtype=t.dtype, device=t.device)
                sel[:, self.idx, 0] = 1.0
                # Broadcast function_vector to (B, 1, H)
                fv = self.function_vector.to(t.dtype).unsqueeze(0)  # (1, 1, H)
                return t + sel * fv.to(t.device)  # (B, S, H)

            if isinstance(output, tuple):
                lst = list(output)
                if isinstance(lst[0], torch.Tensor) and lst[0].dim() == 3:
                    lst[0] = inject(lst[0])
                return tuple(lst)
            else:
                if isinstance(output, torch.Tensor) and output.dim() == 3:
                    return inject(output)
                return output

        target_layer.forward = new_forward
        self._original_forward = original_forward
        self._target_layer = target_layer

    def remove_hooks(self):
        if hasattr(self, '_target_layer') and hasattr(self, '_original_forward'):
            self._target_layer.forward = self._original_forward

        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    # change which layer is edited
    def set_edit_layer(self, new_edit_layer: int):
        """
        Change the layer index where the function vector is injected.
        This restores the previous layer's forward and re-registers hooks
        on the new layer.
        """
        if new_edit_layer == self.edit_layer:
            return

        # Clean up current hooks
        self.remove_hooks()

        # Update layer index
        self.edit_layer = new_edit_layer

        # Register hooks for the new layer
        self._register_hooks()

    # get current edit_layer value
    def get_edit_layer(self) -> int:
        return self.edit_layer

    def forward(self, *args, idx=None, **kwargs):
        if idx is None:
            raise ValueError("You must provide `idx` during the forward pass.")
        self.idx = idx
        return self.model(*args, **kwargs)


def eval_zero_shot_fv(dataset, fv_model, tokenizer, image_processor, processor, model_config, verbose=False):
    # Evaluation
    eval_acc = 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    for i, data in pbar:
        relation = data['relation']
        images = data['images']
        question = data['question']
        answer = data['answer']
        token_id_of_interest = get_answer_id(question, answer, tokenizer, processor, model_config)[0]

        outputs, _ = inference_model(fv_model, tokenizer, model_config, images, question,
                                     image_processor=image_processor, processor=processor, idx=-1)

        output = outputs.logits[:, -1,:].detach().cpu()  # batch_size x n_tokens x vocab_size, only want last token prediction

        if verbose:
            print(question)
            next_token_id = torch.argmax(output, dim=-1)  # shape: [batch]
            target_token_str = tokenizer.decode(token_id_of_interest)
            next_token_str = tokenizer.decode(next_token_id)
            print(f"target tokens: {target_token_str}")
            print(f"next tokens: {next_token_str}")
            print("-" * 60)

        # TRACK probs of tokens of interest
        intervention_rank = compute_individual_token_rank(output, token_id_of_interest)
        matched = intervention_rank < 1
        eval_acc += float(matched)

        # delete any tensors not needed for the next iteration
        del output, outputs
        torch.cuda.empty_cache()

        pbar.set_description("Eval Acc: %.4f" % (eval_acc / (i + 1), ))
    eval_acc = eval_acc / len(dataset)
    return eval_acc


def zero_shot_opt_with_fv_model(train_dataset, test_dataset, epochs: int, fv_model, tokenizer, image_processor, processor, model_config, path=None, lr=1e-2, batch_size=1):
    """
    Optimize the FV on the model using the provided ICL dataset.

    Returns:
    results: dict of topk accuracy on the test dataset, for both the model's n-shot, and n-shot + FV intervention, as well as the token rank of each prediction
    """
    assert batch_size == 1

    optimizer = torch.optim.Adam([fv_model.function_vector], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    optimizer.zero_grad()

    eval_acc = eval_zero_shot_fv(test_dataset, fv_model, tokenizer, image_processor, processor, model_config)
    print(f"Function vector before optimization evaluation accuracy: {eval_acc}.")

    max_acc = 0
    for e in range(epochs):
        tot_loss, n = 0, 0
        indices = torch.randperm(len(train_dataset))
        pbar = tqdm(indices, total=len(train_dataset))
        for i in pbar:
            data = train_dataset[i]
            images = data['images']
            question = data['question']
            answer = data['answer']

            training_prompt, labels, colon_index = construct_training_labels(question, answer, model_config, tokenizer, processor)
            output, _ = inference_model(fv_model, tokenizer, model_config, images, training_prompt,
                                         image_processor=image_processor, processor=processor, labels=labels, idx=colon_index)

            optimizer.zero_grad()
            intervention_nll = output.loss
            tot_loss += intervention_nll.item()

            intervention_nll.backward()
            optimizer.step()

            n += 1
            pbar.set_description(f"Epoch {e + 1} | Loss: {tot_loss / n:.5f}")

        lr_scheduler.step()
        eval_acc = eval_zero_shot_fv(test_dataset, fv_model, tokenizer, image_processor, processor, model_config)
        if eval_acc > max_acc:
            max_acc = eval_acc
            if path is not None:
                torch.save(fv_model.function_vector, path)

    return fv_model.function_vector

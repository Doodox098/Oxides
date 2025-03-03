import matplotlib.pyplot as plt
import torch
from oxides_loss import oxides_loss
import OxideModel
from PIL import Image
import numpy as np
import tqdm


def temperature_shift(delta, max_delta=30, delta_scale=1):
    return max_delta * (2 * torch.nn.functional.sigmoid(delta * delta_scale) - 1)


def draw(oxide_models, global_shift, reference, it, config, begin_losses, usage_losses, residual_losses):
    fig = plt.figure(figsize=(12, 12))
    begin_loss_ax = plt.subplot(2, 3, 1)
    begin_loss_ax.set_title('Begin Losses')
    usage_loss_ax = plt.subplot(2, 3, 2)
    usage_loss_ax.set_title('Usage Losses')
    residual_loss_ax = plt.subplot(2, 3, 3)
    residual_loss_ax.set_title('Residual Losses')
    oxides_ax = plt.subplot(2, 1, 2)
    oxides_ax.set_title('Oxides and residual')
    logs = [f'GLOBAL SHIFT: {global_shift.item()}']

    oxygen, time = reference

    with (torch.no_grad()):
        oxide_models.eval()
        begin_change_loss, begin_loss, usage_loss, residual_loss = \
            oxides_loss(oxide_models, global_shift, reference, it, config)
        begin_losses.append(begin_loss.item() + begin_change_loss.item())
        begin_loss_ax.plot(begin_losses)
        usage_losses.append(usage_loss.item())
        usage_loss_ax.plot(usage_losses)
        residual_losses.append(residual_loss.item())
        residual_loss_ax.plot(residual_losses)
        loss = sum([begin_change_loss, begin_loss, usage_loss, residual_loss])
        time_grid = np.linspace(1400, 2200, num=1000)
        res = torch.zeros(1000).cpu()
        for oxide_name, oxide in oxide_models.items():
            oxide_parameters = "\n\t".join([f"{x[0]} = {x[1].item()}" for x in oxide.named_parameters()])
            logs.append(f'{oxide_name}: {oxide_parameters}')
            ox_res = oxide(torch.tensor(time_grid), global_shift)
            oxides_ax.plot(time_grid, ox_res.detach(), '.', label=oxide_name)
            res += ox_res.cpu()
        oxides_ax.plot(time_grid, res.detach(), '.')
        oxides_ax.plot(time, oxygen, '-')
        oxides_ax.legend()
        oxides_ax.grid(True)
        logs = '\n'.join(logs)
        return fig, logs


def delete_oxides(oxide_models, global_shift, reference, it, config):
    oxygen, time = reference
    delete = []
    delete_after = config['delete_after']

    for oxide_name, oxide in oxide_models.items():
        v_ref = oxygen[np.argmin(np.abs(time - oxide.get_t_max().item() - global_shift.item()))]
        if (
                oxide.get_v_max() < 0.3 * v_ref and it > delete_after) or oxide.get_v_max() < 0.1 * v_ref or oxide.get_v_max() < 0.05:
            delete.append(oxide_name)

    from itertools import combinations
    for oxide_1, oxide_2 in combinations(oxide_models.items(), 2):
        oxide_name_1, oxide_1 = oxide_1
        oxide_name_2, oxide_2 = oxide_2
        if oxide_name_1 in delete or oxide_name_2 in delete:
            continue
        if abs(oxide_1.get_t_max().item() - oxide_2.get_t_max().item()) < 20:
            if abs(oxide_2.get_v_max()) < abs(oxide_1.get_v_max()) or oxide_1 in config['guaranteed_oxides']:
                delete.append(oxide_name_2)
            else:
                delete.append(oxide_name_1)
    for oxide in delete:
        oxide_models.pop(oxide)


def train(oxide_models, global_shift_delta, reference, optimizer, config):
    import matplotlib
    matplotlib.use('TkAgg')
    # Train configuration
    num_epoch = config['num_epoch']
    stop_after = config['stop_after']
    draw_every = config['draw_every']
    show_every = config['show_every']
    delete_after = config['delete_after']
    unstable_after = config['unstable_after']  # Number of iterations to check for monotonicity
    stable_after = config['stable_after']
    window_size = config['window_size']
    frames = []
    begin_losses = []
    residual_losses = []
    usage_losses = []
    # Dictionary to store last K v_max values for each oxide
    oxide_direction_info = {
        oxide_name: {
            "last_direction": None,
            "iter_since_last_change": 0,
            "v_max_history": [],
            "smoothed_v_max": None,
            "is_stable": True,
            "iter_since_last_change_history": [],
        }
        for oxide_name in oxide_models.keys()
    }
    try:
        for it in tqdm.tqdm(range(num_epoch)):
            oxide_models.train()
            global_shift = temperature_shift(global_shift_delta)

            def closure():
                optimizer.zero_grad()

                oxides_to_delete = []
                for oxide_name, oxide in oxide_models.items():
                    if oxide.get_v_max() < 0.01:
                        oxides_to_delete.append(oxide_name)
                for oxide in oxides_to_delete:
                    oxide_models.pop(oxide)

                begin_change_loss, begin_loss, usage_loss, residual_loss = oxides_loss(oxide_models, global_shift,
                                                                                       reference, it, config)
                loss = begin_change_loss + begin_loss + usage_loss + residual_loss
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)
            if it == 200:
                for g in optimizer.param_groups:
                    g['lr'] = 0.01
            if it == 2000:
                for g in optimizer.param_groups:
                    g['momentum'] = 0.9

            # Check for oscillations in v_max
            any_stable = False  # At least one oxide is not oscillating
            for oxide_name, oxide in oxide_models.items():
                current_v_max = oxide.get_v_max().item()
                oxide_direction_info[oxide_name]["v_max_history"].append(current_v_max)

                if len(oxide_direction_info[oxide_name]["v_max_history"]) <= window_size:
                    any_stable = True
                    continue
                oxide_direction_info[oxide_name]["v_max_history"].pop(0)

                current_v_max = np.mean(oxide_direction_info[oxide_name]["v_max_history"])
                prev_v_max = oxide_direction_info[oxide_name].get("smoothed_v_max", None)

                if prev_v_max is not None:
                    # Determine the current direction
                    if current_v_max > prev_v_max:
                        current_direction = "increase"
                    elif current_v_max < prev_v_max:
                        current_direction = "decrease"
                    else:
                        current_direction = oxide_direction_info[oxide_name]["last_direction"]  # No change

                    # Check if direction has changed
                    if current_direction != oxide_direction_info[oxide_name]["last_direction"]:
                        # If direction changed too quickly, it's an oscillation
                        if oxide_direction_info[oxide_name]["iter_since_last_change"] < unstable_after:
                            oxide_direction_info[oxide_name]["is_stable"] = False

                        # Reset the iteration counter
                        oxide_direction_info[oxide_name]["iter_since_last_change"] = 0
                    else:
                        # Increment the iteration counter
                        oxide_direction_info[oxide_name]["iter_since_last_change"] += 1

                    if oxide_direction_info[oxide_name]["iter_since_last_change"] > stable_after:
                        oxide_direction_info[oxide_name]["is_stable"] = True

                    # Update the last direction
                    oxide_direction_info[oxide_name]["last_direction"] = current_direction
                    oxide_direction_info[oxide_name]["iter_since_last_change_history"].append(
                        oxide_direction_info[oxide_name]["iter_since_last_change"])

                # Update the last v_max value
                oxide_direction_info[oxide_name]["smoothed_v_max"] = current_v_max
                if oxide_direction_info[oxide_name]["is_stable"]:
                    any_stable = True

            # Stop training if all oxides are oscillating
            if not any_stable and it > stop_after:
                print(f"Stopping training as all oxides are oscillating. Last Epoch: {it}")
                break
            delete_oxides(oxide_models, global_shift, reference, it, config)
            if it % draw_every == 0:
                fig, logs = draw(oxide_models, global_shift, reference, it, config, begin_losses, usage_losses,
                                 residual_losses)

                fig.canvas.draw()
                image = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')
                frames.append(image)
                if show_every and it % show_every == 0:
                    print(logs)
                    plt.show()
                plt.close('all')

    except Exception as e:
        print(e)
    # fig = plt.figure(figsize=(12, 6))
    # fig.set_dpi(100)
    # for oxide_name, oxide in oxide_models.items():
    #     plt.plot(oxide_direction_info[oxide_name]["iter_since_last_change_history"][-100:], label=oxide_name)
    # plt.ylim([-1, 30])
    # plt.legend()
    # plt.show()
    gif_duration = 10 * 1000
    frames[0].save('train_10_sec.gif', save_all=True, append_images=frames[1:], optimize=False,
                   duration=min(gif_duration / len(frames), 100), loop=0)
    fig, logs = draw(oxide_models, global_shift, reference, it, config, begin_losses, usage_losses,
                                 residual_losses)
    plt.show()
    for oxide_name, oxide in oxide_models.items():
        print(f'{oxide_name}:\n'
              f'\tT_beg_init = {oxide.b_state["init"]} T_beg_final = {oxide.get_t_beg().item()}\n'
              f'\tT_max_init = {oxide.t_state["init"]} T_max_final = {oxide.get_t_max().item()}\n'
              f'\tV_max_init = {oxide.v_state["init"]} V_max_final = {oxide.get_v_max().item()}\n'
              f'\tE_init = {oxide.e_state["init"]} E_final = {oxide.get_E().item()}\n')
import tensorflow as tf
from model import build_pose_net, build_depth_net
from loss import compute_loss


pose_net = build_pose_net()
depth_net = build_depth_net()


@tf.function
def train_step_fun(batch_imgs, step):
    # Each element in the batch contains three consecutive images, stacked along the channels axis:
    # batch_imgs: (batch_size, img_size, img_size, 3 * 3)
    with tf.GradientTape() as tape:
        disp0, disp1, disp2, disp3 = depth_net(batch_imgs[:, :, :, :3])  # Each (batch_size, ?, ?, 1)
        pose_net_output = pose_net(batch_imgs)  # (batch_size, 2 * 6)

        loss = compute_loss(disp0, disp1, disp2, disp3, pose_net_output)



        net_output = model(batch_imgs, training=True)
        loss_value = loss(batch_gt, net_output)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step)
    return loss_value, net_output






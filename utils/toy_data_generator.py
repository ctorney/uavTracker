"""
This program generates a toy dataset for uavTracker detector and tracker.

The images are already in a yolo-consistent shape (multiple of 32),

"""

import os, math, yaml, argparse
import numpy as np
import pandas as pd
from skimage.color import hsv2rgb
import cv2
from collections import deque
from scipy.special import softmax
import colorsys
from utils import init_config


def getRoI(zwk):
    """
    We have to get RoI analytically because otherwise we cannot have overlapping even if we can resolved which recognised contour is which animals bounding box.
    There are two things to do
    TODO make it more reasonable padding for every postion of the ellipse
    TODO make it work with my expanded conciousness (shape i mean physical shape)
    """
    sinzwk = zwk.islong * math.sin(np.pi * zwk.angle / 180)
    coszwk = zwk.islong * math.cos(np.pi * zwk.angle / 180)
    sinzwkw = zwk.iswide * math.sin(np.pi * zwk.angle / 180)
    coszwkw = zwk.iswide * math.cos(np.pi * zwk.angle / 180)
    tail = (int(zwk.x_pos - coszwk), int(zwk.y_pos - sinzwk))
    head = (int(zwk.x_pos + coszwk), int(zwk.y_pos + sinzwk))

    c1 = (int(head[0] + sinzwkw), int(head[1] - coszwkw))
    c2 = (int(head[0] - sinzwkw), int(head[1] + coszwkw))
    c3 = (int(tail[0] - sinzwkw), int(tail[1] + coszwkw))
    c4 = (int(tail[0] + sinzwkw), int(tail[1] - coszwkw))
    offset = 2  # HACK hardcoded offset
    # mins of all
    topleft = (
        np.min([c1[0], c2[0], c3[0], c4[0]]) - offset,
        np.min([c1[1], c2[1], c3[1], c4[1]]) - offset,
    )
    # maxes of all
    bottomright = (
        np.max([c1[0], c2[0], c3[0], c4[0]]) + offset,
        np.max([c1[1], c2[1], c3[1], c4[1]]) + offset,
    )
    return (head, topleft, bottomright)


def updateZwkPosition(zwk, zwks, side):
    """
    Updates position of all Zwierzaks.
    We are allowing them to run on top of each other for now...
    """
    zwk.x_prev = zwk.x_pos
    zwk.y_prev = zwk.y_pos

    cur_pos, is_same_panel = zwk.updatePosition(side)

    zwk.angle = zwk.mm.getDirection()

    zwk.x_pos = int(cur_pos[0])
    zwk.y_pos = int(cur_pos[1])
    return zwk, is_same_panel


class Mooveemodel:
    """
    This movement model need to be just the movement model so my position on the map initis etc have to be moved out of here
    """

    def __init__(
        self,
        x_init,
        y_init,
        mu_s,
        sigma_speed,
        sigma_angular_velocity,
        theta_speed,
        theta_angular_velocity,
    ):
        # [speed and angular velocity]
        self.mu = np.array([mu_s, 0.0])
        self.theta = np.array([theta_speed, theta_angular_velocity])
        self.sigma = np.array([sigma_speed, sigma_angular_velocity])
        self.v = np.array(self.mu)
        self.dt = np.ones(2)
        self.rng = np.random.default_rng()
        self.pos = np.array([x_init, y_init])
        self.angle = 360 * self.rng.uniform()  # prob of going into special state
        self.os = np.array(self.mu)
        self.s = 0
        self.updateSpeed()

    def updateSpeed(self, external_coefficient_of_noise_term=1):
        os1 = self.os
        mu1 = self.mu
        theta1 = self.theta
        dt1 = self.dt
        sigma1 = self.sigma
        rng1 = self.rng

        self.os = (
            os1
            + theta1 * (mu1 - os1) * dt1
            + sigma1
            * [external_coefficient_of_noise_term, 1]
            * rng1.normal(0, np.sqrt(dt1), 2)
        )

        self.angle = self.angle + self.os[1] * dt1[1]
        # self.s = np.log1p(np.exp(self.os[0])) #softplus cause it to get stuck in 0.
        self.s = abs(self.os[0])
        self.v[0] = self.s * np.cos(self.angle)
        self.v[1] = self.s * np.sin(self.angle)

        return self.v

    def getDirection(self):
        return np.degrees(np.arctan2(self.v[1], self.v[0]))

    def prepMovement(self, nsamples=1000):
        """
        Generate quit e a fair bit of datapoints of the movement model
        """
        dd = []
        for i in range(nsamples):
            vv = self.updateSpeed()
            dd.append([i, self.s, self.os[1]])
        df = pd.DataFrame(dd, columns=["time", "speed", "angular_velocity"])
        return df


class Zwierzak:
    """
    Our animal can have different colour or the same
    """

    def __init__(self, zwkid, track_id, x_init, y_init, mm, genmodel, hue=0, sat=1):
        self.mm = mm  # movememnt mode, each animus has its own now
        self.id = zwkid
        self.track_id = (
            track_id  # this changes whenever the animal disappears from view
        )
        self.x_pos = x_init
        self.y_pos = y_init
        self.x_prev = x_init
        self.y_prev = y_init
        self.hsv = (hue, sat, 200)  # initialise as not too-light value
        self.angle = mm.angle
        self.islong = (
            30  # half of width and height as opencv ellipses measurements defined
        )
        self.iswide = 10
        self.speed = 2  # shouldn't that be mu_s?
        self.rng = np.random.default_rng()

        self.genmodel = genmodel
        self.state = 0  # we will use state to define our little accelreated moments.
        self.state_time = 0
        self.external_coefficient_of_noise_term = 1

        # unusual numbers to encourage program loudly crashing
        self.topleft = -111
        self.bottomright = -111
        self.topleft_prev = -111
        self.bottomright_prev = -111

        self.panelswitcher = deque([False, False, False])

    """
    In case of periodic border condition we need to be able to always see our animal. However, if there are multiple animals in the scene it means that their relative position is messed up.
    It is fine though, we are looking at each 3-frame scenario as a separate tracking problem. Also we exclude frames that have animals close to the border.
    """

    def observationPointSwitch(self, is_same_panel):
        self.panelswitcher.popleft()
        self.panelswitcher.append(is_same_panel)
        return np.all(self.panelswitcher)

    """
    every now and then our ALF shrinks and gets a 10x boost of the noise term of the speed that should be visible in the rapid change of position in the next frame
    """

    def updateState(self):
        # for simple model we do not mess with the state
        if self.genmodel == "simple" or self.genmodel == "stochastic":
            return 0

        if self.state == 0:
            if self.rng.uniform() > 0.95:  # prob of going into special state
                self.state = 1
                self.islong = 10
                self.external_coefficient_of_noise_term = 30
                return 0

        if self.state == 1:
            self.state = 0
            self.islong = 20
            self.external_coefficient_of_noise_term = 1
            return 0

    """
    Update the position and tell us if we have moved past the border. Updating position shouldn't really be job of movement model though....?
    """

    def updatePosition(self, side):

        self.updateState()

        new_pos = self.mm.pos + (self.mm.v * self.mm.dt)
        self.mm.pos = new_pos % side
        is_same_panel = True if np.all(new_pos == self.mm.pos) else False

        self.mm.updateSpeed(self.external_coefficient_of_noise_term)
        return self.mm.pos, is_same_panel


class Borders:
    """
    This class shows any natural and unnatural boundaries for the environment
    """

    x_min = 0
    y_min = 0
    x_max = 100
    y_max = 100

    def __init__(
        self, xmi, ymi, xma, yma
    ):  # isn't that a dumb constructor syntax, heh?
        self.x_min = xmi
        self.y_min = ymi
        self.x_max = xma
        self.y_max = yma


def set_alfs(generator_config, setting, mr, side, next_track_id=0):
    genmodel = generator_config[setting]["model"]
    if genmodel == "simple":
        mu_s = generator_config[setting]["mu_s"]
        sigma_speed = generator_config[setting]["sigma_speed"]
        sigma_angular_velocity = generator_config[setting]["sigma_angular_velocity"]
        theta_speed = generator_config[setting]["theta_speed"]
        theta_angular_velocity = generator_config[setting]["theta_angular_velocity"]
    elif genmodel == "stochastic":
        mu_s = np.random.uniform(0, generator_config[setting]["mu_s"])
        sigma_speed = np.random.uniform(0, generator_config[setting]["sigma_speed"])
        sigma_angular_velocity = np.random.uniform(
            0, generator_config[setting]["sigma_angular_velocity"]
        )
        theta_speed = np.random.uniform(0, generator_config[setting]["theta_speed"])
        theta_angular_velocity = np.random.uniform(
            0, generator_config[setting]["theta_angular_velocity"]
        )
    else:
        raise ValueError("Unknown model")

    if "identical" in generator_config[setting]:
        identical = generator_config[setting]["identical"]
    else:
        identical = False

    no_alfs = generator_config[setting]["no_alfs"]
    alfs = []
    next_after_track_id = no_alfs + next_track_id
    for a in range(next_track_id, next_after_track_id):
        x_init, y_init = map(int, map(round, mr.uniform(0, side - 1, 2)))
        mm = Mooveemodel(
            x_init,
            y_init,
            mu_s,
            sigma_speed,
            sigma_angular_velocity,
            theta_speed,
            theta_angular_velocity,
        )
        if identical:
            thue = 0.4
        else:
            thue = mr.uniform(0, 1)

        curalf = Zwierzak(f"alf{a}", a, x_init, y_init, mm, genmodel, hue=thue, sat=1)
        alfs.append(curalf)
    next_track_id = next_after_track_id
    return alfs, next_track_id


def updateAndDrawAlfs(
    toy_label, alf, alfs, side, plane_cur, recthosealfs, img_data, it, next_track_id
):
    alf, is_same_panel = updateZwkPosition(alf, alfs, side)
    if not is_same_panel:
        alf.track_id = next_track_id
        next_track_id += 1
    cv2.ellipse(
        plane_cur,
        (alf.x_pos, alf.y_pos),
        (alf.islong, alf.iswide),
        alf.angle,
        0,
        360,
        colorsys.hsv_to_rgb(alf.hsv[0], alf.hsv[1], alf.hsv[2]),
        -1,
    )
    (head, r1, r2) = getRoI(alf)

    cv2.circle(
        plane_cur, head, 3, (40, 0, 0)
    )  # make alfs head dark as the plane is now light

    roiNotOnBorder = True  # or beyond....
    if (
        r1[0] <= 0
        or r1[0] >= side
        or r2[0] <= 0
        or r2[0] >= side
        or r1[1] <= 0
        or r1[1] >= side
        or r2[1] <= 0
        or r2[1] >= side
    ):
        roiNotOnBorder = False

    recthosealfs.append(alf.observationPointSwitch((is_same_panel and roiNotOnBorder)))

    # uncomment the following line to see bounding boxez
    # DEBUG cv2.rectangle(plane_cur,r1,r2,(123,20,255),2) # show bounding box

    alf.topleft = (float(min(r1[0], r2[0])), float(min(r1[1], r2[1])))
    alf.bottomright = (float(max(r1[0], r2[0])), float(max(r1[1], r2[1])))

    obj = dict()
    obj["name"] = toy_label
    obj["xmin"] = alf.topleft[0]
    obj["ymin"] = alf.topleft[1]
    obj["xmax"] = alf.bottomright[0]
    obj["ymax"] = alf.bottomright[1]
    obj["id"] = alf.id
    obj["time"] = it
    img_data["object"] += [obj]

    return recthosealfs, img_data, next_track_id


def set_name(oname, setting, it):
    return f"{oname}im_{setting}_{it:05d}"


def main(args):

    config = init_config(args)
    generator_config_file = config["synth_generator"]
    with open(generator_config_file, "r") as configfile:
        generator_config = yaml.safe_load(configfile)
    show_img = generator_config["visual"]
    side = (
        int(generator_config["size"]) // 32
    ) * 32  # The generator provides images with annotations so they have to be yolo-compatible size already

    # read from command line
    DEBUG = config["args_debug"]
    print(f"Debugging is {DEBUG}")
    ddir = config["project_directory"]
    os.makedirs(ddir, exist_ok=True)
    oname = config["project_name"]
    preped_images_dir = os.path.join(ddir, config["preped_images_dir"])
    print(f"Only square images work for now!")

    # Prepare a list of when different things happen
    dp_train = config["subsets"]["train"]["number_of_images"]
    dp_test = config["subsets"]["test"]["number_of_images"]

    dp = dp_train + dp_test
    dp_ratio = dp_train / dp

    param_seq_len = generator_config["param_seq_len"]
    stoch_multip = generator_config["datapoints_muliploer_for_stochastic"]
    settings_for_dbtracker_train = generator_config["settings_for_dbtracker_train"]
    settings_for_dbtracker_test = generator_config["settings_for_dbtracker_test"]
    if set(settings_for_dbtracker_train).intersection(set(settings_for_dbtracker_test)):
        raise ValueError(
            "The settings for dbtracker train and test should be disjoint sets"
        )

    settings_for_dbtracker = settings_for_dbtracker_train + settings_for_dbtracker_test
    settings_for_uavtracker = generator_config["settings_for_uavtracker"]
    if set(settings_for_dbtracker).intersection(set(settings_for_uavtracker)):
        raise ValueError(
            "The settings for dbtracker and uavtracker should be disjoint sets"
        )
    all_settings = settings_for_uavtracker + settings_for_dbtracker
    number_of_uavtracker_sets = len(settings_for_uavtracker)
    number_of_dbtracker_sets = len(settings_for_dbtracker)
    dp_per_uavtracker_set = math.ceil(dp / number_of_uavtracker_sets)
    dp_per_dbtracker_set_train = generator_config["datapoints_per_dbtracker_set_train"]
    dp_per_dbtracker_set_test = generator_config["datapoints_per_dbtracker_set_test"]

    # Those are *not* raw images as we forcing them to be yolo-compatible size as they are _already_ annotated!
    # test_dir = os.path.join(ddir,config['raw_imgs_dir'],config['subsets']['test']['directory'])
    # train_dir = os.path.join(ddir,config['raw_imgs_dir'],config['subsets']['train']['directory'])
    test_dir = os.path.join(ddir, config["subsets"]["test"]["directory"])
    train_dir = os.path.join(ddir, config["subsets"]["train"]["directory"])

    # prepare directories
    an_dir = os.path.join(ddir, "annotations")
    gt_dir = os.path.join(ddir, "groundtruths")
    video_dir = os.path.join(ddir, "videos")

    os.makedirs(an_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(preped_images_dir, exist_ok=True)
    annotations_file = an_dir + "/train_data.yml"
    db_train_seq_file = os.path.join(an_dir, config["seq_yml"])
    toy_label = config["common"]["LABELS"][0]
    all_imgs_for_uav = []
    all_db_train_seq = []

    fourCC = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out_test = cv2.VideoWriter(
        os.path.join(video_dir, "test.avi"), fourCC, 5, (side, side), True
    )
    out_train = cv2.VideoWriter(
        os.path.join(video_dir, "train.avi"), fourCC, 5, (side, side), True
    )

    borders = Borders(1, 1, side - 1, side - 1)
    next_track_id = 0

    hdplane = np.zeros((side, side, 3), np.uint8)
    hdplane[:, :, 2] = 255
    hdplane[:, :, 1] = 247
    hdplane[:, :, 0] = 230
    mr = np.random.default_rng()

    x_init, y_init = [side // 2, side // 2]

    for setting in all_settings:
        if setting in settings_for_dbtracker:
            setting_for_dbtracker = True
            train_uav = False
            if setting in settings_for_dbtracker_train:
                train_for_dbtracker = True
                dps = dp_per_dbtracker_set_train
            else:
                train_for_dbtracker = False
                if generator_config[setting]["model"] == "stochastic":
                    dps = dp_per_dbtracker_set_test * stoch_multip
                else:
                    dps = dp_per_dbtracker_set_test

        elif setting in settings_for_uavtracker:
            setting_for_dbtracker = False
            train_for_dbtracker = False
            train_uav = True
            dps = dp_per_uavtracker_set
        else:
            print(f"Warning!!! we are not using settings {settings} for anything")
            pass

        print(
            f"Preparing data with setting {setting} with, train_uav={train_uav} and setting_for_dbtracker={setting_for_dbtracker} with {dps} datapoints"
        )

        # alfs, next_track_id = set_alfs(generator_config, setting, mr, side)
        one_fname_gt = os.path.join(an_dir, f"{setting}.txt")
        video_gt = cv2.VideoWriter(
            os.path.join(video_dir, f"{setting}.avi"), fourCC, 5, (side, side), True
        )
        one_file_gt = open(one_fname_gt, "a")

        for it in range(dps):
            # reset the list of alfs every XXX frames so that long training data has different colours
            if it % param_seq_len == 0:
                alfs, next_track_id = set_alfs(
                    generator_config, setting, mr, side, next_track_id
                )
            plane_cur = hdplane.copy()
            recthosealfs = (
                []
            )  # all animals must be visible and moving within current panel to be useful for training
            save_name_seed = set_name(oname, setting, it)
            save_name = f"{save_name_seed}.jpg"
            img_data = {"object": []}
            img_data["filename"] = save_name
            img_data["width"] = side
            img_data["height"] = side

            for alf in alfs:
                recthosealfs, img_data, next_track_id = updateAndDrawAlfs(
                    toy_label,
                    alf,
                    alfs,
                    side,
                    plane_cur,
                    recthosealfs,
                    img_data,
                    it,
                    next_track_id,
                )

            # saving all the output:
            fnames_gt = os.path.join(gt_dir, f"{save_name_seed}.txt")
            files_gt = open(fnames_gt, "w")

            uav_training_datapoint = it < (dp_ratio * dp_per_uavtracker_set)

            # if DEBUG:
            #     print(recthosealfs)

            # only record the sequence for training images
            # recthosealfs.append(uav_training_datapoint)

            # only record sequence for those images that are used for deebbeast training
            recthosealfs.append((setting_for_dbtracker))
            # recthosealfs.append((setting_for_dbtracker or train_uav))
            # print(recthosealfs)
            record_the_seq = np.all(recthosealfs)

            # recording this sequence for linker training/testing
            if record_the_seq:
                if DEBUG:
                    cv2.putText(
                        plane_cur,
                        "R",
                        (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1.0,
                        (0, 0, 250),
                        2,
                    )
                seq_data = {"object": []}
                seq_data["filename"] = save_name
                p1_fname = set_name(oname, setting, it - 1)
                p2_fname = set_name(oname, setting, it - 2)
                seq_data["p1_filename"] = f"{p1_fname}.jpg"
                seq_data["p2_filename"] = f"{p2_fname}.jpg"

            for alf in alfs:
                if record_the_seq:
                    obj = {}
                    obj["name"] = toy_label
                    obj["xmin"] = alf.topleft[0]
                    obj["ymin"] = alf.topleft[1]
                    obj["xmax"] = alf.bottomright[0]
                    obj["ymax"] = alf.bottomright[1]
                    obj["pxmin"] = alf.topleft_prev[0]
                    obj["pymin"] = alf.topleft_prev[1]
                    obj["pxmax"] = alf.bottomright_prev[0]
                    obj["pymax"] = alf.bottomright_prev[1]
                    seq_data["object"] += [obj]

                one_file_gt.write(save_name + " ")
                one_file_gt.write(str(alf.track_id) + " ")
                one_file_gt.write(str(alf.topleft[0]) + " ")
                one_file_gt.write(str(alf.topleft[1]) + " ")
                one_file_gt.write(str(alf.bottomright[0]) + " ")
                one_file_gt.write(str(alf.bottomright[1]))
                one_file_gt.write("\n")

                files_gt.write(toy_label + " ")
                files_gt.write(str(alf.topleft[0]) + " ")
                files_gt.write(str(alf.topleft[1]) + " ")
                files_gt.write(str(alf.bottomright[0]) + " ")
                files_gt.write(str(alf.bottomright[1]))
                files_gt.write("\n")

                alf.topleft_prev = alf.topleft
                alf.bottomright_prev = alf.bottomright

            files_gt.close()

            video_gt.write(plane_cur)
            # The sequence file for dbtracker is only for training
            if record_the_seq and train_for_dbtracker:
                all_db_train_seq += [seq_data]

            # Just a reminder that our custom setup for uavtraining is a bit weird.
            # Annotations file contains all the training and testing files
            # But depending in which directory the file is, it will be used for testing or training
            if train_uav:
                all_imgs_for_uav += [img_data]
                if uav_training_datapoint:
                    cv2.imwrite(train_dir + "/" + save_name, plane_cur)
                    out_train.write(plane_cur)
                else:
                    cv2.imwrite(test_dir + "/" + save_name, plane_cur)
                    out_test.write(plane_cur)

            # all sequences need to be saved in the preped images dir for MOT sequences
            cv2.imwrite(os.path.join(preped_images_dir, save_name), plane_cur)
            if show_img:
                cv2.imshow("hdplane", plane_cur)
                if DEBUG:
                    key = cv2.waitKey(0)
                else:
                    key = cv2.waitKey(20)
                if key == ord("q"):
                    break

        video_gt.release()
        train_uav = False
        one_file_gt.close()

    with open(annotations_file, "w") as handle:
        yaml.dump(all_imgs_for_uav, handle)
    with open(db_train_seq_file, "w") as handle:
        yaml.dump(all_db_train_seq, handle)

    print("Done and done!")
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate a movement sequence",
        epilog="Any issues and clarifications: github.com/mixmixmix/moovemoo/issues",
    )
    parser.add_argument(
        "--config", "-c", required=True, nargs=1, help="Your yml config file"
    )
    parser.add_argument(
        "--debug", "-d", default=False, action="store_true", help="Debugging"
    )

    args = parser.parse_args()

    main(args)

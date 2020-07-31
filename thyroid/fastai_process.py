from file_cache import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from dynamic_unet.base import *
import uuid
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_df():

    img_file_list = glob('/home/felix/pj/thyroid_seg/input/TNSCUI2020_train/image/*.PNG', recursive=True)

    df = pd.DataFrame({'img_file': img_file_list})

    df['label_path'] = df.img_file.apply(lambda val: val.replace('image', 'mask'))
    df['sn'] = df.img_file.apply(lambda val: os.path.basename(val).split('.')[0]).astype(int)
    df['ID'] = df.img_file.apply(lambda val: os.path.basename(val))
    df = df.sort_values('sn')

    np.random.seed(2007)
    df['fold'] = np.random.randint(0, 5, len(df))
    df['valid'] = df.fold == 4

    # df['valid'] = df.sn%5 >=4
    df.head()

    df['size1'] = df.label_path.apply(lambda val: cv2.imread(val).shape[0])
    df['size2'] = df.label_path.apply(lambda val: cv2.imread(val).shape[1])
    df.sort_values('size1')

    codes = list(range(2))
    print('codes', list(codes))

    df_cat = pd.read_csv('/home/felix/pj/thyroid_seg/input/TNSCUI2020_train/train.csv')
    print(df_cat.shape)
    df_cat.head()

    df = df.merge(df_cat, on=['ID'])
    df.head()
    return df


def get_data(size=224, bs=4):
    codes = list(range(2))
    df = get_df()
    print('size, bs', size, bs)

    src = (SegmentationItemList.from_df(df, path='/', cols='img_file')
           .split_from_df(col='valid')
           .label_from_df(cols='label_path', classes=codes)
           )

    print(len(src.train), len(src.valid), codes)

    # get_transforms()
    data = (src.transform(None, size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))
    test_data = ImageList.from_folder(path="/share/data2/body/thyroid/test/image")
    data.add_test(test_data, tfm_y=False)

    return data


def get_learn(encoder_name, size=224, bs=4, attention=True, wd=1e-2, ):
    data = get_data(size=size, bs=bs)

    def unet_learner(data: DataBunch, pretrained: bool = True, blur_final: bool = True,
                     norm_type: Optional[NormType] = None, split_on: Optional[SplitFuncOrIdxList] = None,
                     blur: bool = False,
                     self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None,
                     last_cross: bool = True,
                     bottle: bool = False, cut: Union[int, Callable] = None, **learn_kwargs: Any) -> Learner:
        #encoder = nn.Sequential(*list(models.densenet121(pretrained=pretrained).children())[0])
        # encoder = nn.Sequential(*list(models.(pretrained=True).children())[0])
        from thyroid.module_encode import UNET_ENCODE
        uet_encoder = UNET_ENCODE.get(encoder_name)()
        unet = DynamicUnet(uet_encoder, n_classes=2, img_size=(size, size), blur=False, blur_final=False,
                           self_attention=attention, y_range=None, norm_type=NormType,
                           last_cross=True,
                           bottle=False)

        learn = Learner(data, unet, **learn_kwargs)
        return learn

    metrics = [dice,  partial(dice, iou=True)]
    learn = unet_learner(data, metrics=metrics, wd=wd)
    return learn


if __name__ == '__main__':
    for encode in ['densenet201', 'densenet121']: #$(hostname)
        for attention in [True, False,  ]:
            for epoch in range(15, 30, 5):
                encode = encode
                attention = attention
                lr = 1e-4
                bs = 4
                size = 224
                #epoch = 10
                rand = str(uuid.uuid1())[:7]
                print(f'bs={bs}, encode={encode}, attention={attention}, lr={lr}, size={size}, rand:{rand}, epoch:{epoch}')
                learn = get_learn(encode, size=size, bs=bs, attention=attention)

                print('===='*10)
                pd.Series(learn.data.valid_ds.items).to_csv(f'./output/valid_{rand}_item.csv')
                pd.Series(learn.data.test_ds.items).to_csv(f'./output/test_{rand}_item.csv')
                print('====' * 20)

                learn.fit_one_cycle(epoch, slice(lr), pct_start=0.9)
                res_test, _ = learn.get_preds(DatasetType.Test)
                res_valid, _ = learn.get_preds(DatasetType.Valid)

                os.makedirs('./output', exist_ok=True)
                np.save(f'./output/test_{rand}_{encode}_{epoch}_{attention}_{size}', res_test)
                np.save(f'./output/valid_{rand}_{encode}_{epoch}_{attention}_{size}', res_valid)


                del learn
                gc.collect()

       # learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
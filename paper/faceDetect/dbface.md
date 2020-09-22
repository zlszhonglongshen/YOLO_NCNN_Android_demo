# dbface源码解读

## 1、主函数
```
class App(object):
    def __init__(self, labelfile, imagesdir):

        self.width, self.height = 800, 800 #输入图片大小
        self.mean = [0.408, 0.447, 0.47] #
        self.std = [0.289, 0.274, 0.278] #图片进行归一化处理
        self.batch_size = 18 #bs大小
        self.lr = 1e-4 #初始学习率
        self.gpus = [2] #[0, 1, 2, 3] #GPU id
        self.gpu_master = self.gpus[0]
        self.model = DBFace(has_landmark=True, wide=64, has_ext=True, upmode="UCBA") #DBface主题框架
        self.model.init_weights() #初始权重
        self.model = nn.DataParallel(self.model, device_ids=self.gpus) 
        self.model.cuda(device=self.gpu_master) #gpu训练
        self.model.train() #

        self.focal_loss = losses.FocalLoss()
        self.giou_loss = losses.GIoULoss()
        self.landmark_loss = losses.WingLoss(w=2)
        self.train_dataset = LDataset(labelfile, imagesdir, mean=self.mean, std=self.std, width=self.width, height=self.height)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24) #加载数据
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) #优化函数
        self.per_epoch_batchs = len(self.train_loader)
        self.iter = 0
        self.epochs = 150


    def set_lr(self, lr):

        self.lr = lr
        log.info(f"setting learning rate to: {lr}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr #设置不同学习率


    def train_epoch(self, epoch):
        
        for indbatch, (images, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, num_objs, keep_mask) in enumerate(self.train_loader):

            self.iter += 1

            batch_objs = sum(num_objs)
            batch_size = self.batch_size

            if batch_objs == 0:
                batch_objs = 1

            heatmap_gt          = heatmap_gt.to(self.gpu_master)
            heatmap_posweight   = heatmap_posweight.to(self.gpu_master)
            keep_mask           = keep_mask.to(self.gpu_master)
            reg_tlrb            = reg_tlrb.to(self.gpu_master)
            reg_mask            = reg_mask.to(self.gpu_master)
            landmark_gt         = landmark_gt.to(self.gpu_master)
            landmark_mask       = landmark_mask.to(self.gpu_master)
            images              = images.to(self.gpu_master)

            hm, tlrb, landmark  = self.model(images)
            hm = hm.sigmoid()
            hm = torch.clamp(hm, min=1e-4, max=1-1e-4)
            tlrb = torch.exp(tlrb)

            hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight, keep_mask=keep_mask) / batch_objs
            reg_loss = self.giou_loss(tlrb, reg_tlrb, reg_mask)*5
            landmark_loss = self.landmark_loss(landmark, landmark_gt, landmark_mask)*0.1
            loss = hm_loss + reg_loss + landmark_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_flt = epoch + indbatch / self.per_epoch_batchs

            if indbatch % 10 == 0:
                log.info(
                    f"iter: {self.iter}, lr: {self.lr:g}, epoch: {epoch_flt:.2f}, loss: {loss.item():.2f}, hm_loss: {hm_loss.item():.2f}, "
                    f"box_loss: {reg_loss.item():.2f}, lmdk_loss: {landmark_loss.item():.5f}"
                )

            if indbatch % 1000 == 0:
                log.info("save hm")
                hm_image = hm[0, 0].cpu().data.numpy()
                common.imwrite(f"{jobdir}/imgs/hm_image.jpg", hm_image * 255)
                common.imwrite(f"{jobdir}/imgs/hm_image_gt.jpg", heatmap_gt[0, 0].cpu().data.numpy() * 255)

                image = np.clip((images[0].permute(1, 2, 0).cpu().data.numpy() * self.std + self.mean) * 255, 0, 255).astype(np.uint8)
                outobjs = eval_tool.detect_images_giou_with_netout(hm, tlrb, landmark, threshold=0.1, ibatch=0)

                im1 = image.copy()
                for obj in outobjs:
                    common.drawbbox(im1, obj)
                common.imwrite(f"{jobdir}/imgs/train_result.jpg", im1)



    def train(self):

        lr_scheduer = {
            1: 1e-3,
            2: 2e-3,
            3: 1e-3,
            60: 1e-4,
            120: 1e-5
        }

        # train
        self.model.train()
        for epoch in range(self.epochs):

            if epoch in lr_scheduer:
                self.set_lr(lr_scheduer[epoch])

            self.train_epoch(epoch)
            file = f"{jobdir}/models/{epoch + 1}.pth"
            common.mkdirs_from_file_path(file)
            torch.save(self.model.module.state_dict(), file)

```

## 数据加载主函数、
### 解析原始文件
```python
def parse_facials_webface(facials):

    ts = []
    for facial in facials:
        x, y, w, h = facial[:4] 
        box = [x, y, x + w - 1, y + h - 1]
        landmarks = None

        if w * h < 4 * 4: #舍弃掉人脸面积较小的
            continue

        if len(facial) >= 19:
            landmarks = []
            for i in range(5):
                x, y, t = facial[i * 3 + 4:i * 3 + 4 + 3]
                if t == -1: #人脸关键点不符合规定
                    landmarks = None
                    break

                landmarks.append([x, y])

        ts.append(BBox(label="facial", xyrb=box, landmark=landmarks, rotate = False))
    return ts

```
### 数据增强


##损失函数





# Image Quality

## Repositories with image quality metrics
1. [PIQ](https://github.com/photosynthesis-team/piq)
2. [sewar](https://vk.com/away.php?to=https%3A%2F%2Fgithub.com%2Fandrewekhalel%2Fsewar&el=snippet)
3. [VQMT](https://github.com/rolinh/VQMT)
4. [FOVQA](https://github.com/Scholles007/Framework-for-Objective-Visual-Quality-Assessment-FOVQA)
5. [Image Quality Tools](https://github.com/sattarab/image-quality-tools)

## Goal
Создать `MVP` приложения, которое вычисляет и отображает метрики, выявляющие различия между двумя изображениями.

## Tasks 
- [ ] Create functions `get_dataset` and `get_dataloader` for loading images from dataset TID2013
- [ ] Create function `get_metrics` for calculating image quality metrics
- [ ] Test piq metrics on TID2013 dataset
- [ ] Test sewar metrics on TID2013 dataset
- [ ] Test VQMT metrics on TID2013 dataset
- [ ] Test FOVQA metrics on TID2013 dataset
- [ ] Test Image Quality Tools metrics on TID2013 dataset
- [ ] Compare metrics on TID2013 dataset

## Notes
> Для тестирования loss функций я решил использовать dataset `TID2013`.
> Сделал загрузку dataset'а в оперативную память, так как он не очень большой и это ускорит работу с ним.
> Для этого создал класс `TID2013Dataset`.
> Попытался учесть ошибки, которые могут возникнуть при загрузке данных.
> Dataset занимает `~7Gb` оперативной памяти.

Репозиторий с MVP: [Image Quality]()
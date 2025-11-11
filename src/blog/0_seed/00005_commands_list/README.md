---
{
  "title": "Commands list for basic things I usually forget",
  "authors": ["Aryan V S"],
  "date": "2025-11-02",
  "tags": ["linux", "slurm", "miscellaneous"]
}
---

- Print simple cluster information
```bash
sinfo -o "%N %c %m %G"

NODELIST CPUS MEMORY GRES
cluster2qh-nodeset1-[0-3] 224 2933888 gpu:8
```

- Launch debugging single node job
```bash
srun --gres=gpu:8 --cpus-per-task=32 --mem=2000G --pty bash -i
```

- Upload checkpoints to aws at a specific interval (here, 500 steps)
```bash
for d in ckpt_step*; do     step=$(echo "$d" | grep -o '[0-9]\+' | sed 's/^0*//');     if (( step % 500 == 0 )); then         aws s3 cp --recursive "$d"         s3://morpheusai-data/aryan/lora/teacher-forcing/1p3b---lr-1e-4---wd-1e-5---ema-0p99---rank-256/"$d";     fi; done
```

- Read/write/random I/O test:
```bash
sudo apt install -y fio
fio --name=rwtest --rw=randrw --size=1G --bs=4k --numjobs=4 --runtime=60 --directory=/path/to/volume --group_reporting
```

- Modify docker storage location (overlayfs cannot be done on NFS storage, so make sure ext4)
```bash
sudo systemctl stop docker docker.socket containerd
# Edit /etc/docker/daemon.json to have {"data-root": "/scratch/local/.docker"} (basically, point it to ext4 system)
sudo systemctl start docker docker.socket containerd
docker info -f '{{ .DockerRootDir}}'
```

- Build docker container from Dockerfile
```bash
sudo docker build --progress=tty -t cu128-pytorch28:latest .
```

- Importing image to enroot
```bash
export CUR_TMPDIR=$TMPDIR
export TMPDIR=/scratch/shared/.tmp-enroot
export ENROOT_DATA_PATH=/scratch/shared/.enroot
export ENROOT_CACHE_PATH=/scratch/shared/.enroot/cache
export ENROOT_RUNTIME_PATH=/scratch/shared/.enroot/runtime
mkdir -p $TMPDIR $ENROOT_DATA_PATH $ENROOT_CACHE_PATH $ENROOT_RUNTIME_PATH
enroot import docker://<USERNAME>@<ORGANIZATION>/cu128-pytorch28:latest
export TMPDIR=$CUR_TMPDIR
```

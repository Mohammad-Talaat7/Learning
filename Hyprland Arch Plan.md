# **Before Installation**
- update pacman databases with:
```bash
pacman -Sy
```

- install pipewire and enable its service as it causing an error during installation
```bash
pacman -S pipewire wireplumber pipewire-pulse pipewire-alsa

systemctl --user enable --now pipewire.service wireplumber.service pipewire-pulse.service 
```

- install updated version of `archinstall`
```bash
pacman -S archinstall
```

- install arch hyprland desktop
```bash

```
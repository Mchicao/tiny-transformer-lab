# WSL en D: — traslado verificado

El 2026-10-01 se trasladaron Ubuntu y los discos Docker WSL a D:, por autorización explícita del usuario. Se conservan respaldos completos y configuraciones originales. La CLI Colab está operativa y el login sigue pendiente.

## Resultado

| Elemento | Destino y validación |
| --- | --- |
| Ubuntu | `D:\WSL\Ubuntu\ext4.vhdx`; registro WSL actualizado por `--manage --move`, UID 1000 y `/home/matias` conservados. Arranque y CLI 0.7.4 verificados. |
| Docker WSL | `D:\WSL\Docker\main\ext4.vhdx` y `disk\docker_data.vhdx`. Junction desde la ruta original en C:; configuración y registro Docker conservados. La distribución `docker-desktop` arrancó mediante esa ruta y se devolvió al estado detenido. |
| Swap | `.wslconfig` apunta a `D:\\WSL\\Swap\\swap.vhdx`; archivo creado en D: y swap activo de 2 GiB observado en `/proc/swaps`. No se cambiaron sus límites ni las demás opciones. |
| Rollback | `D:\WSL\Backups\wsl-move-20261001-002740`, con los tres discos, `.wslconfig` y settings Docker originales. |

Antes del traslado ambos WSL y Docker Desktop estaban detenidos. Se respaldaron los discos y se verificó igualdad exacta SHA-256. Tras moverlos se repitió la comprobación antes de arrancarlos; los archivos originales de los discos ya no ocupan C:. Los cambios normales de un filesystem al arrancar no se comparan con un hash anterior al arranque.

Los valores de espacio libre antes/después están en `output/wsl-move-20261001-002740/verification.json`. Se liberaron aproximadamente 62,7 GiB en C:. D: conserva tanto los discos activos como otra copia para rollback.

No se arrancó el motor Docker ni se verificaron contenedores: la comprobación cubre conservación de los discos, enlace y arranque de la distribución WSL. Tampoco se inició OAuth, asignó GPU ni entrenó modelos.

## Recuperación si hiciera falta

Las rutas originales constan en `output/wsl-move-20261001-002740/plan.json`. Antes de cualquier recuperación, detener las distribuciones y volver a respaldar su estado actual; no sobrescribir actividad posterior con el snapshot anterior.

- Ubuntu puede volver a su ubicación original con el mismo mecanismo `wsl --manage Ubuntu --move`, conservando su registro y usuario. No requiere `--unregister`.
- Para Docker, conservar los discos actuales y retirar sólo la junction antes de restaurar la carpeta original. No borrar recursivamente la ruta del enlace: sus archivos activos están en D:.
- La configuración original de swap se conserva como `wslconfig.original`. Restaurarla requiere reiniciar la VM WSL para aplicar el cambio.

No se ha ejecutado rollback; los respaldos sirven como ruta de recuperación y no como prueba de una restauración completa.

## Login Colab

Ya se puede seguir el [login interactivo de la CLI](../guides/colab-cli-wsl.md#próxima-autorización). El comando `usage` consulta el saldo tras OAuth y no crea runtimes. Los permisos de la CLI deben revisarse antes de aceptar el consentimiento.

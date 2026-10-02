# Usar Colab CLI desde Windows y WSL

La CLI oficial `google-colab-cli` **0.7.4** está instalada en Ubuntu 24.04.4 / WSL2, con Python 3.12.3. Autenticación, saldo, T4, ejecución y transferencia de checkpoints reales fueron verificados el 2026-10-01. Las sesiones utilizadas ya no están disponibles; consultar estado antes de reutilizar sus nombres.

## Abrir la CLI desde PowerShell

```powershell
Set-Location D:\Proyectos_B\tiny-transformer-lab
.\scripts\agents\colab-wsl.ps1 version
.\scripts\agents\colab-wsl.ps1 --help
```

La primera orden devuelve `Version: 0.7.4`. El lanzador transmite argumentos y el código de salida a PowerShell.

Desde una terminal Ubuntu:

```sh
~/.local/share/tiny-transformer-lab/colab-cli/colab version
~/.local/share/tiny-transformer-lab/colab-cli/colab --help
```

No se añadió un comando global a `PATH` ni se modificaron perfiles de shell. D: no está montado en esta distribución; el lanzador funciona sin modificar ese montaje.

## Almacenamiento WSL trasladado a D:

Traslado autorizado por el usuario y verificado el 2026-10-01:

- Ubuntu: `D:\WSL\Ubuntu\ext4.vhdx`, movido mediante `wsl --manage Ubuntu --move`. Usuario Linux y rutas internas conservados.
- Docker WSL: `D:\WSL\Docker\main\ext4.vhdx` y `D:\WSL\Docker\disk\docker_data.vhdx`. La ruta original `C:\Users\matia\AppData\Local\Docker\wsl` es una junction a `D:\WSL\Docker`; contiene un enlace, no otra copia de los discos. La configuración Docker no cambió.
- Swap: `D:\WSL\Swap\swap.vhdx`. Se añadió únicamente `swapFile` a `.wslconfig`, conservando las otras opciones.
- Respaldos completos privados: `D:\WSL\Backups\wsl-move-20261001-002740`. Se conservan; no borrarlos como parte del uso de la CLI.

Los tres discos coincidieron por SHA-256 con sus respaldos antes del arranque. Ubuntu y la distribución interna Docker arrancaron después del traslado; la CLI siguió mostrando `Version: 0.7.4`. WSL creó el swap en D: y Ubuntu lo muestra activo. No se arrancó Docker Desktop ni se ejecutaron contenedores para esta comprobación.

Evidencia y rutas originales: `output/wsl-move-20261001-002740/`. El traslado conserva el HOME privado de la CLI dentro del disco Ubuntu, ahora almacenado físicamente en D:.

## Aislamiento y reproducibilidad

| Elemento | Ubicación |
| --- | --- |
| Entorno y lanzador Linux | `/home/matias/.local/share/tiny-transformer-lab/colab-cli/` |
| Python aislado | `.venv/`, creado con `/usr/bin/python3.12` existente |
| HOME de la CLI | `home/` dentro de esa carpeta; credenciales, estado y logs propios de la herramienta quedan separados de otros proyectos |
| Caché uv de instalación | `cache/uv/` dentro de esa carpeta |
| Paquete fijado | `configs/requirements-colab-cli.in` |
| Dependencias con hashes | `configs/requirements-colab-cli.lock.txt`, copiado a Linux como `requirements.lock.txt` |
| Lanzadores versionables | `scripts/agents/colab-wsl.sh` y `scripts/agents/colab-wsl.ps1` |

Se resolvieron e instalaron 52 paquetes desde PyPI, sólo wheels, con versiones fijadas, hashes obligatorios y fecha de corte `2026-09-27`. No se instaló PyTorch ni un stack de entrenamiento en WSL. La carpeta completa ocupa aproximadamente **244 MiB**, incluyendo caché; no es una medición de RAM.

El lanzador usa un HOME privado y `umask 077`. Las rutas `~` dentro de la CLI corresponden a ese HOME privado; para archivos externos, usar rutas explícitas. No se importan cookies ni credenciales del navegador.

## Pruebas realizadas

- `version`, `--help` y ayudas de `upload`, `download`, `usage` y `drivemount`: código cero.
- Comando inexistente: código 2 propagado correctamente por PowerShell.
- `uv pip check`: 52 paquetes compatibles.
- Sintaxis PowerShell y shell válida; hashes de launcher y lockfile comparados con sus copias Linux.
- Sin `token.json` ni archivo de sesiones de la CLI; no se inició autenticación, creó runtime o consumió GPU mediante esta instalación.

Evidencia: `output/colab-cli-install-20261001/verification.json`, `logs/app/colab-cli-install-20261001.log` y `logs/tests/colab-cli-wsl-*-20261001.log`.

La resolución del lockfile en Windows descargó un CPython 3.12.3 auxiliar administrado por uv. Ese intérprete no se utilizó para la instalación Linux y no reemplazó el `.venv` AMD de Windows.

## Próxima autorización

La CLI necesita su propio consentimiento OAuth; la sesión de Helium no la autentica automáticamente. El siguiente paso sería autenticar la cuenta elegida y consultar saldo con `usage`, sin crear GPU ni entrenar.

Después del traslado ya se puede realizar ese login desde una terminal PowerShell interactiva:

```powershell
.\scripts\agents\colab-wsl.ps1 --auth oauth2 usage
```

Abrir la URL que imprime la CLI en un navegador, seleccionar la cuenta del plan y revisar el consentimiento. Pegar el código que muestra Google exclusivamente en la terminal. La versión instalada solicita identidad, Colab, `drive.file` y `cloud-platform`; si no se desean esos permisos, detenerse antes de confirmar. El login todavía no fue ejecutado por el agente.

`colab auth` autentica una VM para servicios GCP: no es el comando de inicio de sesión inicial de la CLI. No ejecutar los ejemplos de `new`/`run` de la documentación oficial como comprobación de instalación: asignan recursos reales.

La instalación no resuelve ni demuestra el montaje Drive. La prohibición de entrenar continúa vigente hasta una instrucción explícita.

## Estado tras el experimento autorizado

La cuenta `el.matias.ch@gmail.com` ya está autenticada mediante OAuth oficial. No se automatizaron contraseñas ni 2FA ni se extrajeron cookies. Aunque el consentimiento dejó Google Cloud sin seleccionar, el token efectivo reportó también `cloud-platform` al comprobarlo con `whoami`; no asumir scopes reducidos sólo por el estado de la pantalla. No se usaron servicios GCP.

El usuario autorizó posteriormente el sanity Colab 3M/AdamW. Ver [resultado y límites](../technical/primer-entrenamiento-colab.md). Login válido no garantiza disponibilidad GPU ni persistencia de un runtime.

El lanzador PowerShell admite stdin. Ejemplo sin entrenamiento, con una sesión propia disponible:

```powershell
'print("Conexión verificada")' | .\scripts\agents\colab-wsl.ps1 exec -s MI_SESION --timeout 60
```

La ruta UNC `\\wsl.localhost\Ubuntu\...` no funcionó para recuperar archivos. Las copias grandes se transfirieron como bytes mediante `wsl.exe ... python3`, evitando convertir binarios en texto; se verificó SHA-256 antes de extraerlos en D:.

Fuente: [CLI oficial de Google Colab](https://github.com/googlecolab/google-colab-cli).

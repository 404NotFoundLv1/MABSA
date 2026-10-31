# Backup destinations

The approved code and full 15-seed evaluation evidence were pushed to:

- Repository: https://github.com/404NotFoundLv1/MABSA
- Branch: `backup/main-base-20261031`
- Frozen initial backup commit: `89f844bd964540f10da9d390459eff879344095a`
- Tag: `main-base-v1-20261031`

The user subsequently requested that weights be downloaded to Windows instead of
uploaded to GitHub. GitHub weight upload was interrupted before the first archive
completed. The two fully verified source archives remain in
`/home/whj/reproject/mact_backup_20261031/` on the Linux server.

The requested Windows destination is `D:\PythonProjects\checkpoints`.
The agent's server session cannot access this Windows drive. Run
`download-to-windows.ps1` in Windows PowerShell to transfer both archives over SSH
and check their SHA-256 values. It defaults to `whj@10.16.81.144`, port 22, as
observed in the active server SSH connection. Supply `-Server` and `-Port` if your
Windows SSH connection uses another host alias or forwarded port.

The script keeps existing files, skips already verified archives, and preserves
failed partial downloads. It prompts through normal SSH authentication; no
password or access token is stored in this repository.

Windows transfer has not been verified by the agent. The initial packaging
manifest's remote-backup flag describes the state before the Git push; weights
are intentionally excluded from Git.

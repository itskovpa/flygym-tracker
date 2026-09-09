# GitHub communication

The user requires the authenticated GitHub connector for GitHub communication, especially publishing commits, updating branches, and creating or updating pull requests. Do not retry command-line pushes through Windows Credential Manager or switch credentials to plaintext. In this Windows sandbox, Credential Manager cannot access its credential store and GitHub CLI configuration access is denied.

Local Git remains suitable for status, diffs, tests, commits, and backups. Read-only fetch/ls-remote over public HTTPS worked and can be used to synchronize or verify connector updates without invoking authentication.

For publishing local changes with the connector, create the required blobs/trees/commits and update the intended ref without force. Verify each uploaded tree hash against the intended local snapshot before updating the branch. Preserve original local commits on a backup branch if connector-created commit IDs differ. Fetch the published branch and synchronize local history only after checking identical trees and a clean working directory. Preserve any new local work and concurrent remote changes.

If the GitHub connector is unavailable or reports an authentication error, report that specific blocker; do not fall back to the known-broken Windows Credential Manager path. Follow the user's existing authorization for writes; this method does not authorize unsolicited messages, merges, or other unrelated actions.


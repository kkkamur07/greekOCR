"""Password hashing via the bcrypt algorithm (not encryption).

We use the ``bcrypt`` library directly (bcrypt 4.x; passlib is not used).

- **Algorithm**: bcrypt (adaptive one-way password hash)
- **Salt / cost**: ``bcrypt.gensalt()`` → default **12 rounds** (2^12 iterations)
- **Stored form**: ASCII string like ``$2b$12$...`` (version, cost, salt, hash)
- **Verification**: ``bcrypt.checkpw`` (constant-time compare of hash)

Passwords are never stored in plaintext; only the bcrypt hash is persisted in ``users.hashed_password``.
"""

import bcrypt

# bcrypt default work factor when using gensalt() without rounds=
BCRYPT_ROUNDS = 12


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

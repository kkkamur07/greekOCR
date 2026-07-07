import { useState, type ChangeEventHandler } from 'react';

type PasswordInputProps = {
  id: string;
  value: string;
  onChange: ChangeEventHandler<HTMLInputElement>;
  autoComplete: 'current-password' | 'new-password';
  placeholder?: string;
  minLength?: number;
  'aria-describedby'?: string;
  required?: boolean;
};

function EyeIcon({ open }: { open: boolean }) {
  if (open) {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M3 3l18 18M10.58 10.58A2 2 0 0 0 12 15a2 2 0 0 0 1.42-.58M9.88 5.09A10.94 10.94 0 0 1 12 5c5.52 0 10 4.48 10 7a11.11 11.11 0 0 1-1.67 2.17M6.11 6.11A10.94 10.94 0 0 0 2 12c0 2.52 4.48 7 10 7 1.74 0 3.37-.37 4.82-1.02"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M2 12s4.48-7 10-7 10 7 10 7-4.48 7-10 7-10-7-10-7Z"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.75" />
    </svg>
  );
}

export function PasswordInput({
  id,
  value,
  onChange,
  autoComplete,
  placeholder = '••••••••',
  minLength,
  'aria-describedby': ariaDescribedBy,
  required = true,
}: PasswordInputProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="password-input">
      <input
        id={id}
        type={visible ? 'text' : 'password'}
        autoComplete={autoComplete}
        placeholder={placeholder}
        aria-describedby={ariaDescribedBy}
        minLength={minLength}
        required={required}
        value={value}
        onChange={onChange}
      />
      <button
        type="button"
        className="password-input__toggle"
        aria-label={visible ? 'Hide password' : 'Show password'}
        aria-pressed={visible}
        onClick={() => setVisible((current) => !current)}
      >
        <EyeIcon open={visible} />
      </button>
    </div>
  );
}

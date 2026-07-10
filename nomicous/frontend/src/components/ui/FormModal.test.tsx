import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { FormModal } from './FormModal';

describe('FormModal', () => {
  it('moves focus into the dialog, traps Tab, and restores the opener', async () => {
    const onClose = vi.fn();
    render(
      <>
        <button type="button">Open dialog</button>
        <FormModal
          open
          title="Create project"
          onClose={onClose}
          onSubmit={(event) => event.preventDefault()}
          submitLabel="Create"
        >
          <label htmlFor="project-name">Name</label>
          <input id="project-name" />
        </FormModal>
      </>,
    );

    const opener = screen.getByRole('button', { name: 'Open dialog' });
    opener.focus();
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveFocus());

    const submit = screen.getByRole('button', { name: 'Create' });
    submit.focus();
    fireEvent.keyDown(submit, { key: 'Tab' });
    expect(screen.getByLabelText('Name')).toHaveFocus();
    fireEvent.keyDown(screen.getByLabelText('Name'), { key: 'Tab', shiftKey: true });
    expect(submit).toHaveFocus();

    fireEvent.keyDown(screen.getByLabelText('Name'), { key: 'Escape' });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('restores focus after the dialog closes', async () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <>
        <button type="button">Open dialog</button>
        <FormModal
          open={false}
          title="Create project"
          onClose={onClose}
          onSubmit={(event) => event.preventDefault()}
          submitLabel="Create"
        >
          <label htmlFor="project-name">Name</label>
          <input id="project-name" />
        </FormModal>
      </>,
    );
    const opener = screen.getByRole('button', { name: 'Open dialog' });
    opener.focus();
    rerender(
      <>
        <button type="button">Open dialog</button>
        <FormModal
          open
          title="Create project"
          onClose={onClose}
          onSubmit={(event) => event.preventDefault()}
          submitLabel="Create"
        >
          <label htmlFor="project-name">Name</label>
          <input id="project-name" />
        </FormModal>
      </>,
    );
    await waitFor(() => expect(screen.getByLabelText('Name')).toHaveFocus());

    rerender(
      <>
        <button type="button">Open dialog</button>
        <FormModal
          open={false}
          title="Create project"
          onClose={onClose}
          onSubmit={(event) => event.preventDefault()}
          submitLabel="Create"
        >
          <label htmlFor="project-name">Name</label>
          <input id="project-name" />
        </FormModal>
      </>,
    );

    expect(screen.getByRole('button', { name: 'Open dialog' })).toHaveFocus();
  });
});

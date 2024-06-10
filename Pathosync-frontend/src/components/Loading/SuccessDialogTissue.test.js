import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import '@testing-library/jest-dom/extend-expect';
import SuccessDialogTissue from './SuccessDialogTissue';

// Mock the useNavigate hook from react-router-dom
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: jest.fn(),
}));

describe('SuccessDialogTissue', () => {
  const message = 'Test Message';
  const accuracy = '95%';
  const loss = '0.05';
  const onDone = jest.fn();
  const onProceed = jest.fn();
  const onRetrain = jest.fn();
  const navigate = require('react-router-dom').useNavigate;

  beforeEach(() => {
    navigate.mockReset();
    onDone.mockReset();
    onProceed.mockReset();
    onRetrain.mockReset();
  });

  test('renders component with correct message, accuracy, and loss', () => {
    render(
      <MemoryRouter>
        <SuccessDialogTissue
          message={message}
          accuracy={accuracy}
          loss={loss}
          onDone={onDone}
          onProceed={onProceed}
          onRetrain={onRetrain}
        />
      </MemoryRouter>
    );

    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getByText(`Accuracy: ${accuracy}`)).toBeInTheDocument();
    expect(screen.getByText(`Loss: ${loss}`)).toBeInTheDocument();
  });

  test('calls onDone when Done button is clicked', () => {
    render(
      <MemoryRouter>
        <SuccessDialogTissue
          message={message}
          accuracy={accuracy}
          loss={loss}
          onDone={onDone}
          onProceed={onProceed}
          onRetrain={onRetrain}
        />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByText('Done'));
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  test('calls onProceed and navigates when Proceed button is clicked', () => {
    render(
      <MemoryRouter>
        <SuccessDialogTissue
          message={message}
          accuracy={accuracy}
          loss={loss}
          onDone={onDone}
          onProceed={onProceed}
          onRetrain={onRetrain}
        />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByText('Proceed'));
    expect(onProceed).toHaveBeenCalledTimes(1);
    expect(navigate).toHaveBeenCalledWith('/Predictions');
  });

  test('calls onRetrain when Retrain button is clicked', () => {
    render(
      <MemoryRouter>
        <SuccessDialogTissue
          message={message}
          accuracy={accuracy}
          loss={loss}
          onDone={onDone}
          onProceed={onProceed}
          onRetrain={onRetrain}
        />
      </MemoryRouter>
    );

    fireEvent.click(screen.getByText('Retrain'));
    expect(onRetrain).toHaveBeenCalledTimes(1);
  });
});

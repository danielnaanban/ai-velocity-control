# Backend Integration Plan

The goal is to connect the React frontend to a live Render backend, removing all mock data and ensuring robust error handling and environment-based configuration.

## Scope
- Configure environment variables for the API URL.
- Update the API client/hooks to use the live backend.
- Remove all local mock data logic.
- Implement loading states and user-friendly error messages.
- Update `.gitignore` to protect environment files.

## Affected Areas
- **Root**: `.env` (create), `.gitignore` (update)
- **Frontend**: `src/App.tsx`, `src/hooks/use-trading-data.ts` (and any other files with fetch logic)
- **Backend (User-side)**: User must update CORS on Render.

## Phases

### Phase 1: Environment Configuration
- Create `.env` in the root directory.
- Add `VITE_API_URL=https://ai-velocity-backend.onrender.com`.
- Append `.env` and `.env.local` to `.gitignore`.
- **Owner**: `quick_fix_engineer`

### Phase 2: Refactor API Logic & Remove Mocks
- Update `src/hooks/use-trading-data.ts` to use `import.meta.env.VITE_API_URL`.
- Modify `handleGenerateSignal` in `src/App.tsx` (or the corresponding hook) to call the specific endpoint: `POST ${API_URL}/api/suggestions/refresh`.
- Remove any fallback to mock data or hardcoded signals.
- **Owner**: `frontend_engineer`

### Phase 3: UI/UX Enhancements (Loading & Error States)
- Ensure the "GENERATE AI SIGNAL" button is disabled when `isLoading` is true.
- Wrap fetch calls in `try/catch`.
- If a fetch fails, display a "sonner" toast or UI text: "AI Bot Offline. Check backend." instead of generic errors.
- **Owner**: `frontend_engineer`

### Phase 4: Validation & Instructions
- Verify that no hardcoded `http://localhost` strings remain in `src/`.
- Provide clear deployment steps for the user.
- **Owner**: `quick_fix_engineer`

## Execution Handoff

**Plan status:** ready

**Dispatch order:**
1. quick_fix_engineer — Initialize env and gitignore.
2. frontend_engineer — Implement API logic and UI states.
3. quick_fix_engineer — Cleanup and final verification.

**Per-agent instructions:**

### 1. quick_fix_engineer
- **Phases:** 1
- **Scope:** Create `.env` with `VITE_API_URL=https://ai-velocity-backend.onrender.com`. Update `.gitignore` to include `.env` and `.env.local`.
- **Files:** `.env`, `.gitignore`
- **Depends on:** none
- **Acceptance criteria:** `.env` exists with correct content; `.gitignore` contains `.env`.

### 2. frontend_engineer
- **Phases:** 2, 3
- **Scope:** 
  - Update `src/hooks/use-trading-data.ts` to use `import.meta.env.VITE_API_URL`.
  - Ensure `fetchSignal` calls `POST /api/suggestions/refresh`.
  - Remove all mock data logic from `src/hooks/use-trading-data.ts` and `src/App.tsx`.
  - Implement `try/catch` in `fetchSignal` and show toast "AI Bot Offline. Check backend." on failure.
  - Verify `isLoading` disables the trigger button.
- **Files:** `src/hooks/use-trading-data.ts`, `src/App.tsx`
- **Depends on:** Phase 1
- **Acceptance criteria:** No mock data in code. Clicking button triggers POST request to the live API. UI shows specific error message on failure.

### 3. quick_fix_engineer
- **Phases:** 4
- **Scope:** Search project for any remaining hardcoded `localhost` or `http://` URLs and replace with `import.meta.env.VITE_API_URL` where appropriate.
- **Files:** `src/**/*`
- **Depends on:** Phase 2
- **Acceptance criteria:** Zero hardcoded backend URLs in `src/`.

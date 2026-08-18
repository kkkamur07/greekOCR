"use client";

import { Suspense } from "react";
import { RegisterPage } from "../../pages/RegisterPage";

export default function RegisterRoute() {
  return (
    <Suspense fallback={null}>
      <RegisterPage />
    </Suspense>
  );
}

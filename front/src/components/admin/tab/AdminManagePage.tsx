'use client';

import { useState } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Button } from 'primereact/button';
import AdminCreateModal from './AdminCreateModal';

interface AdminUser {
  id: number;
  name: string;
  email: string;
  role: string;
}

export default function AdminManagePage() {
  const [admins, setAdmins] = useState<AdminUser[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleAddAdmin = (newAdmin: { email: string; name: string; role: string }) => {
    setAdmins((prev) => [
      ...prev,
      { id: prev.length + 1, ...newAdmin },
    ]);
  };

  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow p-4 mb-4 flex justify-between items-center">
        <h2 className="text-lg font-semibold">관리자 계정 목록</h2>
        <Button
          label="관리자 추가"
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 border-none"
          onClick={() => setIsModalOpen(true)}
        />
      </div>

      <div className="bg-white rounded-lg shadow">
        <DataTable value={admins} paginator rows={5} stripedRows>
          <Column field="id" header="번호" />
          <Column field="name" header="관리자 이름" />
          <Column field="email" header="관리자 아이디" />
          <Column field="role" header="권한" />
        </DataTable>
      </div>

      <AdminCreateModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onCreate={handleAddAdmin}
      />
    </div>
  );
}

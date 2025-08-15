'use client';

import { useState } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Button } from 'primereact/button';
import AdminCreateModal from './AdminCreateModal';
import { useQuery } from '@tanstack/react-query';
import { getAdmins } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';

export default function AdminManagePage() {
  const { token } = useAuth();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // --- API로 관리자 목록 조회 ---
  const { data: adminData, isLoading } = useQuery({
    queryKey: ['admins'],
    queryFn: () => {
      if (!token) return null;
      return getAdmins(token, 0, 10, true); // 활성 관리자만 조회
    },
    enabled: !!token,
  });

    return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow p-4 mb-4 flex justify-between items-center">
        <h2 className="text-lg font-semibold">관리자 계정 목록</h2>
        <Button
          label="관리자 추가"
          icon="pi pi-plus"
          onClick={() => setIsModalOpen(true)}
        />
      </div>

      <div className="bg-white rounded-lg shadow">
        <DataTable value={adminData?.content} loading={isLoading} paginator rows={10} stripedRows>
          <Column field="adminId" header="관리자 아이디" />
          <Column field="adminName" header="관리자 이름" />
          <Column field="role" header="권한" />
          <Column field="createdAt" header="생성일" body={(rowData) => new Date(rowData.createdAt).toLocaleDateString()} />
        </DataTable>
      </div>

      <AdminCreateModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}

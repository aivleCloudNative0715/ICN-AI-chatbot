'use client';

import { useState } from 'react';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Button } from 'primereact/button';

interface AdminUser {
  id: number;
  name: string;
  email: string;
  role: string;
}

export default function AdminManagePage() {
  const [admins, setAdmins] = useState<AdminUser[]>(
    Array.from({ length: 20 }, (_, i) => ({
      id: i + 1,
      name: '김00',
      email: 'abc@admin.com',
      role: '관리자',
    }))
  );

  return (
    <div className="p-6">
      <div className="bg-white rounded-lg shadow p-4 mb-4 flex justify-between items-center">
        <h2 className="text-lg font-semibold">관리자 계정 목록</h2>
        <Button
          label="관리자 추가"
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 border-none"
        />
      </div>

      <div className="bg-white rounded-lg shadow">
        <DataTable
          value={admins}
          paginator
          rows={10}
          className="p-datatable-sm"
          paginatorTemplate="PrevPageLink PageLinks NextPageLink"
          currentPageReportTemplate="{first} - {last} of {totalRecords}"
          rowsPerPageOptions={[5, 10, 20]}
          stripedRows
        >
          <Column
            field="id"
            header="번호"
            body={(rowData, { rowIndex }) => rowIndex + 1}
            className="text-center w-20"
          />
          <Column field="name" header="관리자 이름" className="text-center" />
          <Column field="email" header="관리자 아이디" className="text-center" />
          <Column field="role" header="권한" className="text-center" />
        </DataTable>
      </div>
    </div>
  );
}

// src/app/privacy/page.tsx
import React from 'react';

export default function PrivacyPolicyPage() {
  const serviceName = "인천공항 AI 챗봇"; // 서비스 이름

  return (
    <div className="max-w-4xl mx-auto p-8 bg-white my-8 rounded-lg shadow">
      <h1 className="text-3xl font-bold mb-6 border-b pb-4">{serviceName} 개인정보 처리방침</h1>
      <div className="prose max-w-none">
        <p className="mb-4">
          {serviceName}은(는) 「개인정보 보호법」 제30조에 따라 정보주체의 개인정보를 보호하고 이와 관련한 고충을 신속하고 원활하게 처리할 수 있도록 하기 위하여 다음과 같이 개인정보 처리방침을 수립·공개합니다.
        </p>

        <h2 className="text-xl font-semibold mt-6">제1조 (개인정보의 처리 목적)</h2>
        <p>
          회사는 다음의 목적을 위하여 개인정보를 처리합니다. 처리하고 있는 개인정보는 다음의 목적 이외의 용도로는 이용되지 않으며, 이용 목적이 변경되는 경우에는 별도의 동의를 받는 등 필요한 조치를 이행할 예정입니다.
        </p>
        <ul className="list-disc pl-5">
          <li><strong>홈페이지 회원 가입 및 관리:</strong> 회원 가입의사 확인, 회원자격 유지·관리, 서비스 부정이용 방지, 고충처리 등</li>
          <li><strong>서비스 제공:</strong> AI 챗봇 서비스 제공, 문의/건의 게시판 운영 등</li>
        </ul>

        <h2 className="text-xl font-semibold mt-6">제2조 (처리하는 개인정보의 항목)</h2>
        <p>회사는 다음의 개인정보 항목을 처리하고 있습니다.</p>
        <ul className="list-disc pl-5">
          <li><strong>필수항목:</strong> 아이디, 비밀번호, 이메일 주소</li>
          <li><strong>자동수집항목:</strong> IP주소, 쿠키, 서비스 이용기록, 방문기록 등</li>
        </ul>
        
        <h2 className="text-xl font-semibold mt-6">제3조 (개인정보의 처리 및 보유 기간)</h2>
        <p>
          회사는 법령에 따른 개인정보 보유·이용기간 또는 정보주체로부터 개인정보를 수집 시에 동의받은 개인정보 보유·이용기간 내에서 개인정보를 처리·보유합니다.
        </p>
        <ul className="list-disc pl-5">
          <li><strong>홈페이지 회원 정보:</strong> 회원 탈퇴 후 30일까지. 다만, 관계 법령 위반에 따른 수사·조사 등이 진행 중인 경우에는 해당 수사·조사 종료 시까지 보유합니다.</li>
        </ul>

        <h2 className="text-xl font-semibold mt-6">제4조 (개인정보의 파기)</h2>
        <p>
          회사는 개인정보 보유기간의 경과, 처리목적 달성 등 개인정보가 불필요하게 되었을 때에는 지체없이 해당 개인정보를 파기합니다. 전자적 파일 형태는 기록을 재생할 수 없는 기술적 방법을 사용하며, 종이 문서는 분쇄하거나 소각하여 파기합니다.
        </p>
        
        {/* KISA 가이드라인의 필수 항목(제3자 제공, 위탁, 정보주체 권리 등)을 참고하여 추가 작성 */}

      </div>
    </div>
  );
}
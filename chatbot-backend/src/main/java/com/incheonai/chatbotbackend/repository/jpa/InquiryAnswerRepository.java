package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.InquiryAnswer;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface InquiryAnswerRepository extends JpaRepository<InquiryAnswer, Integer> {
    List<InquiryAnswer> findAllByInquiryId(Integer inquiryId);
}

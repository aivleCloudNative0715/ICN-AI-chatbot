package com.incheonai.chatbotbackend.repository.secondary;

import com.incheonai.chatbotbackend.domain.mongodb.Atmos;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface AtmosRepository extends MongoRepository<Atmos, String> {

    Optional<Atmos> findFirstByOrderByTimeDesc();
}
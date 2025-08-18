// com/incheonai/chatbotbackend/controller/AirportInfoController.java
package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.TemperatureResponseDto;
import com.incheonai.chatbotbackend.dto.external.*;
import com.incheonai.chatbotbackend.service.AirportInfoService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.List;

@RestController
@RequestMapping("/api/airport")
public class AirportInfoController {

    private final AirportInfoService airportInfoService;

    public AirportInfoController(AirportInfoService airportInfoService) {
        this.airportInfoService = airportInfoService;
    }

    @GetMapping("/parking")
    public Mono<ResponseEntity<List<ParkingInfoItem>>> getParkingInfo() {
        return airportInfoService.getParkingInfo()
                .map(ResponseEntity::ok);
    }

//    @GetMapping("/weather")
//    public Mono<ResponseEntity<List<ArrivalWeatherInfoItem>>> getArrivalsWeatherInfo() {
//        return airportInfoService.getArrivalsWeatherInfo()
//                .map(ResponseEntity::ok);
//    }

    @GetMapping("/weather")
    public Mono<ResponseEntity<TemperatureResponseDto>> getLatestWeather() {
        return airportInfoService.getLatestTemperature()
                .map(ResponseEntity::ok);
    }

    @GetMapping("/forecast")
    public Mono<ResponseEntity<List<PassengerForecastItem>>> getPassengerForecast(
            @RequestParam(defaultValue = "0") String selectDate
    ) {
        return airportInfoService.getPassengerForecast(selectDate)
                .map(ResponseEntity::ok);
    }

    @GetMapping("/flights/arrivals")
    public Mono<ResponseEntity<List<FlightArrivalInfoItem>>> getFlightArrivals(@RequestParam(required = false) String flightId) {
        return airportInfoService.getFlightArrivalsInfo(flightId)
                .map(ResponseEntity::ok);
    }

    @GetMapping("/flights/departures")
    public Mono<ResponseEntity<List<FlightDepartureInfoItem>>> getFlightDepartures(@RequestParam(required = false) String flightId) {
        return airportInfoService.getFlightDeparturesInfo(flightId)
                .map(ResponseEntity::ok);
    }
}
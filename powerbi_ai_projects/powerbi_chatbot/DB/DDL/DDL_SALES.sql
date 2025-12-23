--Sales Dashboard
CREATE DATABASE sales_ab;
USE sales_ab;

CREATE TABLE sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    region VARCHAR(50),
    amount DECIMAL(10,2),
    date DATE
);

INSERT INTO sales (region, amount, date) VALUES
('Muscat', 1000.00, '2023-10-01'),
('Dhofar', 1500.00, '2023-10-01'),
('Muscat', 1200.00, '2023-09-01');
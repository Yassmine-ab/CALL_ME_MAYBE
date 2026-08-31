---
title: "Notes - Call Me Maybe"
author: "Yassmine"
date: "28/08/2026"
---

# Principe de base d'un LLM
Un LLM (Large Language Model, grand modèle de langage en français) est un modèle d'intelligence artificielle entraîné sur d'énormes quantités de texte, ce qui lui permet de comprendre et de générer du langage humain. Concrètement, il a appris à prédire, à chaque étape, le token le plus statistiquement probable pour continuer une séquence de texte donnée.

## Qu'est-ce que le Function Calling ?
Le function calling est une technique où, au lieu de répondre en langage naturel, le LLM produit une sortie structurée — un nom de fonction et ses arguments typés — que d'autres programmes peuvent directement exécuter afin de déclencher des actions concrètes.

## Qu'est-ce que le format JSON ?
Le JSON, c'est un format d'écriture pour organiser des données, pensé pour être facile à lire à la fois par un humain et par une machine. C'est le format que le LLM utilise pour "parler" aux programmes dans le function calling. Sa syntaxe s'organise autour de paires clé-valeur, entre accolades {} :
json
{
  "name": "Yassmine",
  "42 student": true
}

## Qu'est-ce que le Constrained Decoding ?
Le constrained decoding (décodage contraint en français) est une technique qui permet à un petit modèle de générer du JSON valide de manière fiable (99%+), alors qu'il aurait normalement produit des erreurs de syntaxe fatales pour un format aussi contraignant que le JSON.
L'idée est de restreindre, à chaque étape, la liste des tokens autorisés en fonction de la structure attendue - au lieu de laisser le LLM choisir librement le token le plus probable, quitte à casser la syntaxe JSON
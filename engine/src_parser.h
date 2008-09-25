/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     EOL = 258,
     INTNUM = 259,
     DBLNUM = 260,
     SYMBOL = 261,
     STRING = 262,
     NIL = 263,
     END = 264,
     DEF = 265,
     WHILE = 266,
     BREAK = 267,
     CONTINUE = 268,
     DROPPING = 269,
     IF = 270,
     ELSE = 271,
     ELIF = 272,
     FOR = 273,
     FORFIRST = 274,
     FORLAST = 275,
     FORMIDDLE = 276,
     SWITCH = 277,
     CASE = 278,
     DEFAULT = 279,
     SELECT = 280,
     SENDER = 281,
     SELF = 282,
     GIVE = 283,
     TRY = 284,
     CATCH = 285,
     RAISE = 286,
     CLASS = 287,
     FROM = 288,
     OBJECT = 289,
     RETURN = 290,
     GLOBAL = 291,
     LAMBDA = 292,
     INIT = 293,
     LOAD = 294,
     LAUNCH = 295,
     CONST_KW = 296,
     ATTRIBUTES = 297,
     PASS = 298,
     EXPORT = 299,
     IMPORT = 300,
     DIRECTIVE = 301,
     COLON = 302,
     FUNCDECL = 303,
     STATIC = 304,
     INNERFUNC = 305,
     FORDOT = 306,
     LISTPAR = 307,
     LOOP = 308,
     ENUM = 309,
     TRUE_TOKEN = 310,
     FALSE_TOKEN = 311,
     OUTER_STRING = 312,
     CLOSEPAR = 313,
     OPENPAR = 314,
     CLOSESQUARE = 315,
     OPENSQUARE = 316,
     DOT = 317,
     ARROW = 318,
     VBAR = 319,
     ASSIGN_POW = 320,
     ASSIGN_SHL = 321,
     ASSIGN_SHR = 322,
     ASSIGN_BXOR = 323,
     ASSIGN_BOR = 324,
     ASSIGN_BAND = 325,
     ASSIGN_MOD = 326,
     ASSIGN_DIV = 327,
     ASSIGN_MUL = 328,
     ASSIGN_SUB = 329,
     ASSIGN_ADD = 330,
     OP_EQ = 331,
     OP_TO = 332,
     COMMA = 333,
     QUESTION = 334,
     OR = 335,
     AND = 336,
     NOT = 337,
     LE = 338,
     GE = 339,
     LT = 340,
     GT = 341,
     NEQ = 342,
     EEQ = 343,
     PROVIDES = 344,
     OP_NOTIN = 345,
     OP_IN = 346,
     HASNT = 347,
     HAS = 348,
     DIESIS = 349,
     ATSIGN = 350,
     CAP_CAP = 351,
     VBAR_VBAR = 352,
     AMPER_AMPER = 353,
     MINUS = 354,
     PLUS = 355,
     PERCENT = 356,
     SLASH = 357,
     STAR = 358,
     POW = 359,
     SHR = 360,
     SHL = 361,
     TILDE = 362,
     NEG = 363,
     AMPER = 364,
     BANG = 365,
     DECREMENT = 366,
     INCREMENT = 367,
     DOLLAR = 368
   };
#endif
/* Tokens.  */
#define EOL 258
#define INTNUM 259
#define DBLNUM 260
#define SYMBOL 261
#define STRING 262
#define NIL 263
#define END 264
#define DEF 265
#define WHILE 266
#define BREAK 267
#define CONTINUE 268
#define DROPPING 269
#define IF 270
#define ELSE 271
#define ELIF 272
#define FOR 273
#define FORFIRST 274
#define FORLAST 275
#define FORMIDDLE 276
#define SWITCH 277
#define CASE 278
#define DEFAULT 279
#define SELECT 280
#define SENDER 281
#define SELF 282
#define GIVE 283
#define TRY 284
#define CATCH 285
#define RAISE 286
#define CLASS 287
#define FROM 288
#define OBJECT 289
#define RETURN 290
#define GLOBAL 291
#define LAMBDA 292
#define INIT 293
#define LOAD 294
#define LAUNCH 295
#define CONST_KW 296
#define ATTRIBUTES 297
#define PASS 298
#define EXPORT 299
#define IMPORT 300
#define DIRECTIVE 301
#define COLON 302
#define FUNCDECL 303
#define STATIC 304
#define INNERFUNC 305
#define FORDOT 306
#define LISTPAR 307
#define LOOP 308
#define ENUM 309
#define TRUE_TOKEN 310
#define FALSE_TOKEN 311
#define OUTER_STRING 312
#define CLOSEPAR 313
#define OPENPAR 314
#define CLOSESQUARE 315
#define OPENSQUARE 316
#define DOT 317
#define ARROW 318
#define VBAR 319
#define ASSIGN_POW 320
#define ASSIGN_SHL 321
#define ASSIGN_SHR 322
#define ASSIGN_BXOR 323
#define ASSIGN_BOR 324
#define ASSIGN_BAND 325
#define ASSIGN_MOD 326
#define ASSIGN_DIV 327
#define ASSIGN_MUL 328
#define ASSIGN_SUB 329
#define ASSIGN_ADD 330
#define OP_EQ 331
#define OP_TO 332
#define COMMA 333
#define QUESTION 334
#define OR 335
#define AND 336
#define NOT 337
#define LE 338
#define GE 339
#define LT 340
#define GT 341
#define NEQ 342
#define EEQ 343
#define PROVIDES 344
#define OP_NOTIN 345
#define OP_IN 346
#define HASNT 347
#define HAS 348
#define DIESIS 349
#define ATSIGN 350
#define CAP_CAP 351
#define VBAR_VBAR 352
#define AMPER_AMPER 353
#define MINUS 354
#define PLUS 355
#define PERCENT 356
#define SLASH 357
#define STAR 358
#define POW 359
#define SHR 360
#define SHL 361
#define TILDE 362
#define NEG 363
#define AMPER 364
#define BANG 365
#define DECREMENT 366
#define INCREMENT 367
#define DOLLAR 368




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 61 "/Users/gniccolai/Progetti/falcon/core/engine/src_parser.yy"
lex_value_t {
   Falcon::int64 integer;
   Falcon::numeric numeric;
   char * charp;
   Falcon::String *stringp;
   Falcon::Symbol *symbol;
   Falcon::Value *fal_val;
   Falcon::Expression *fal_exp;
   Falcon::Statement *fal_stat;
   Falcon::ArrayDecl *fal_adecl;
   Falcon::DictDecl *fal_ddecl;
   Falcon::SymbolList *fal_symlist;
   Falcon::List *fal_genericList;
}
/* Line 1529 of yacc.c.  */
#line 290 "/Users/gniccolai/Progetti/falcon/core/engine/src_parser.hpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif




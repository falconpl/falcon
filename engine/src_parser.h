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
     FORALL = 276,
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
     COLON = 300,
     FUNCDECL = 301,
     STATIC = 302,
     FORDOT = 303,
     LOOP = 304,
     CLOSEPAR = 305,
     OPENPAR = 306,
     CLOSESQUARE = 307,
     OPENSQUARE = 308,
     DOT = 309,
     ASSIGN_POW = 310,
     ASSIGN_SHL = 311,
     ASSIGN_SHR = 312,
     ASSIGN_BXOR = 313,
     ASSIGN_BOR = 314,
     ASSIGN_BAND = 315,
     ASSIGN_MOD = 316,
     ASSIGN_DIV = 317,
     ASSIGN_MUL = 318,
     ASSIGN_SUB = 319,
     ASSIGN_ADD = 320,
     ARROW = 321,
     FOR_STEP = 322,
     OP_TO = 323,
     COMMA = 324,
     QUESTION = 325,
     OR = 326,
     AND = 327,
     NOT = 328,
     LET = 329,
     LE = 330,
     GE = 331,
     LT = 332,
     GT = 333,
     NEQ = 334,
     EEQ = 335,
     OP_EQ = 336,
     OP_ASSIGN = 337,
     PROVIDES = 338,
     OP_NOTIN = 339,
     OP_IN = 340,
     HASNT = 341,
     HAS = 342,
     DIESIS = 343,
     ATSIGN = 344,
     CAP = 345,
     VBAR = 346,
     AMPER = 347,
     MINUS = 348,
     PLUS = 349,
     PERCENT = 350,
     SLASH = 351,
     STAR = 352,
     POW = 353,
     SHR = 354,
     SHL = 355,
     BANG = 356,
     NEG = 357,
     DECREMENT = 358,
     INCREMENT = 359,
     DOLLAR = 360
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
#define FORALL 276
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
#define COLON 300
#define FUNCDECL 301
#define STATIC 302
#define FORDOT 303
#define LOOP 304
#define CLOSEPAR 305
#define OPENPAR 306
#define CLOSESQUARE 307
#define OPENSQUARE 308
#define DOT 309
#define ASSIGN_POW 310
#define ASSIGN_SHL 311
#define ASSIGN_SHR 312
#define ASSIGN_BXOR 313
#define ASSIGN_BOR 314
#define ASSIGN_BAND 315
#define ASSIGN_MOD 316
#define ASSIGN_DIV 317
#define ASSIGN_MUL 318
#define ASSIGN_SUB 319
#define ASSIGN_ADD 320
#define ARROW 321
#define FOR_STEP 322
#define OP_TO 323
#define COMMA 324
#define QUESTION 325
#define OR 326
#define AND 327
#define NOT 328
#define LET 329
#define LE 330
#define GE 331
#define LT 332
#define GT 333
#define NEQ 334
#define EEQ 335
#define OP_EQ 336
#define OP_ASSIGN 337
#define PROVIDES 338
#define OP_NOTIN 339
#define OP_IN 340
#define HASNT 341
#define HAS 342
#define DIESIS 343
#define ATSIGN 344
#define CAP 345
#define VBAR 346
#define AMPER 347
#define MINUS 348
#define PLUS 349
#define PERCENT 350
#define SLASH 351
#define STAR 352
#define POW 353
#define SHR 354
#define SHL 355
#define BANG 356
#define NEG 357
#define DECREMENT 358
#define INCREMENT 359
#define DOLLAR 360




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union
#line 66 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
}
/* Line 1489 of yacc.c.  */
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.hpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif




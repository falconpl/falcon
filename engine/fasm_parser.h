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
     NAME = 259,
     COMMA = 260,
     COLON = 261,
     DENTRY = 262,
     DGLOBAL = 263,
     DVAR = 264,
     DCONST = 265,
     DATTRIB = 266,
     DLOCAL = 267,
     DPARAM = 268,
     DFUNCDEF = 269,
     DENDFUNC = 270,
     DFUNC = 271,
     DMETHOD = 272,
     DPROP = 273,
     DPROPREF = 274,
     DCLASS = 275,
     DCLASSDEF = 276,
     DCTOR = 277,
     DENDCLASS = 278,
     DFROM = 279,
     DEXTERN = 280,
     DMODULE = 281,
     DLOAD = 282,
     DSWITCH = 283,
     DSELECT = 284,
     DCASE = 285,
     DENDSWITCH = 286,
     DLINE = 287,
     DSTRING = 288,
     DISTRING = 289,
     DCSTRING = 290,
     DHAS = 291,
     DHASNT = 292,
     DINHERIT = 293,
     DINSTANCE = 294,
     SYMBOL = 295,
     EXPORT = 296,
     LABEL = 297,
     INTEGER = 298,
     REG_A = 299,
     REG_B = 300,
     REG_S1 = 301,
     REG_S2 = 302,
     REG_L1 = 303,
     REG_L2 = 304,
     NUMERIC = 305,
     STRING = 306,
     STRING_ID = 307,
     TRUE_TOKEN = 308,
     FALSE_TOKEN = 309,
     I_LD = 310,
     I_LNIL = 311,
     NIL = 312,
     I_ADD = 313,
     I_SUB = 314,
     I_MUL = 315,
     I_DIV = 316,
     I_MOD = 317,
     I_POW = 318,
     I_ADDS = 319,
     I_SUBS = 320,
     I_MULS = 321,
     I_DIVS = 322,
     I_POWS = 323,
     I_INC = 324,
     I_DEC = 325,
     I_INCP = 326,
     I_DECP = 327,
     I_NEG = 328,
     I_NOT = 329,
     I_RET = 330,
     I_RETV = 331,
     I_RETA = 332,
     I_FORK = 333,
     I_PUSH = 334,
     I_PSHR = 335,
     I_PSHN = 336,
     I_POP = 337,
     I_LDV = 338,
     I_LDVT = 339,
     I_STV = 340,
     I_STVR = 341,
     I_STVS = 342,
     I_LDP = 343,
     I_LDPT = 344,
     I_STP = 345,
     I_STPR = 346,
     I_STPS = 347,
     I_TRAV = 348,
     I_TRAN = 349,
     I_TRAL = 350,
     I_IPOP = 351,
     I_XPOP = 352,
     I_GENA = 353,
     I_GEND = 354,
     I_GENR = 355,
     I_GEOR = 356,
     I_JMP = 357,
     I_IFT = 358,
     I_IFF = 359,
     I_BOOL = 360,
     I_EQ = 361,
     I_NEQ = 362,
     I_GT = 363,
     I_GE = 364,
     I_LT = 365,
     I_LE = 366,
     I_UNPK = 367,
     I_UNPS = 368,
     I_CALL = 369,
     I_INST = 370,
     I_SWCH = 371,
     I_SELE = 372,
     I_NOP = 373,
     I_TRY = 374,
     I_JTRY = 375,
     I_PTRY = 376,
     I_RIS = 377,
     I_LDRF = 378,
     I_ONCE = 379,
     I_BAND = 380,
     I_BOR = 381,
     I_BXOR = 382,
     I_BNOT = 383,
     I_MODS = 384,
     I_AND = 385,
     I_OR = 386,
     I_ANDS = 387,
     I_ORS = 388,
     I_XORS = 389,
     I_NOTS = 390,
     I_HAS = 391,
     I_HASN = 392,
     I_GIVE = 393,
     I_GIVN = 394,
     I_IN = 395,
     I_NOIN = 396,
     I_PROV = 397,
     I_END = 398,
     I_PEEK = 399,
     I_PSIN = 400,
     I_PASS = 401,
     I_SHR = 402,
     I_SHL = 403,
     I_SHRS = 404,
     I_SHLS = 405,
     I_LDVR = 406,
     I_LDPR = 407,
     I_LSB = 408,
     I_INDI = 409,
     I_STEX = 410,
     I_TRAC = 411,
     I_WRT = 412,
     I_STO = 413
   };
#endif
/* Tokens.  */
#define EOL 258
#define NAME 259
#define COMMA 260
#define COLON 261
#define DENTRY 262
#define DGLOBAL 263
#define DVAR 264
#define DCONST 265
#define DATTRIB 266
#define DLOCAL 267
#define DPARAM 268
#define DFUNCDEF 269
#define DENDFUNC 270
#define DFUNC 271
#define DMETHOD 272
#define DPROP 273
#define DPROPREF 274
#define DCLASS 275
#define DCLASSDEF 276
#define DCTOR 277
#define DENDCLASS 278
#define DFROM 279
#define DEXTERN 280
#define DMODULE 281
#define DLOAD 282
#define DSWITCH 283
#define DSELECT 284
#define DCASE 285
#define DENDSWITCH 286
#define DLINE 287
#define DSTRING 288
#define DISTRING 289
#define DCSTRING 290
#define DHAS 291
#define DHASNT 292
#define DINHERIT 293
#define DINSTANCE 294
#define SYMBOL 295
#define EXPORT 296
#define LABEL 297
#define INTEGER 298
#define REG_A 299
#define REG_B 300
#define REG_S1 301
#define REG_S2 302
#define REG_L1 303
#define REG_L2 304
#define NUMERIC 305
#define STRING 306
#define STRING_ID 307
#define TRUE_TOKEN 308
#define FALSE_TOKEN 309
#define I_LD 310
#define I_LNIL 311
#define NIL 312
#define I_ADD 313
#define I_SUB 314
#define I_MUL 315
#define I_DIV 316
#define I_MOD 317
#define I_POW 318
#define I_ADDS 319
#define I_SUBS 320
#define I_MULS 321
#define I_DIVS 322
#define I_POWS 323
#define I_INC 324
#define I_DEC 325
#define I_INCP 326
#define I_DECP 327
#define I_NEG 328
#define I_NOT 329
#define I_RET 330
#define I_RETV 331
#define I_RETA 332
#define I_FORK 333
#define I_PUSH 334
#define I_PSHR 335
#define I_PSHN 336
#define I_POP 337
#define I_LDV 338
#define I_LDVT 339
#define I_STV 340
#define I_STVR 341
#define I_STVS 342
#define I_LDP 343
#define I_LDPT 344
#define I_STP 345
#define I_STPR 346
#define I_STPS 347
#define I_TRAV 348
#define I_TRAN 349
#define I_TRAL 350
#define I_IPOP 351
#define I_XPOP 352
#define I_GENA 353
#define I_GEND 354
#define I_GENR 355
#define I_GEOR 356
#define I_JMP 357
#define I_IFT 358
#define I_IFF 359
#define I_BOOL 360
#define I_EQ 361
#define I_NEQ 362
#define I_GT 363
#define I_GE 364
#define I_LT 365
#define I_LE 366
#define I_UNPK 367
#define I_UNPS 368
#define I_CALL 369
#define I_INST 370
#define I_SWCH 371
#define I_SELE 372
#define I_NOP 373
#define I_TRY 374
#define I_JTRY 375
#define I_PTRY 376
#define I_RIS 377
#define I_LDRF 378
#define I_ONCE 379
#define I_BAND 380
#define I_BOR 381
#define I_BXOR 382
#define I_BNOT 383
#define I_MODS 384
#define I_AND 385
#define I_OR 386
#define I_ANDS 387
#define I_ORS 388
#define I_XORS 389
#define I_NOTS 390
#define I_HAS 391
#define I_HASN 392
#define I_GIVE 393
#define I_GIVN 394
#define I_IN 395
#define I_NOIN 396
#define I_PROV 397
#define I_END 398
#define I_PEEK 399
#define I_PSIN 400
#define I_PASS 401
#define I_SHR 402
#define I_SHL 403
#define I_SHRS 404
#define I_SHLS 405
#define I_LDVR 406
#define I_LDPR 407
#define I_LSB 408
#define I_INDI 409
#define I_STEX 410
#define I_TRAC 411
#define I_WRT 412
#define I_STO 413




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif




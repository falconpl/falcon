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
     DIMPORT = 283,
     DSWITCH = 284,
     DSELECT = 285,
     DCASE = 286,
     DENDSWITCH = 287,
     DLINE = 288,
     DSTRING = 289,
     DISTRING = 290,
     DCSTRING = 291,
     DHAS = 292,
     DHASNT = 293,
     DINHERIT = 294,
     DINSTANCE = 295,
     SYMBOL = 296,
     EXPORT = 297,
     LABEL = 298,
     INTEGER = 299,
     REG_A = 300,
     REG_B = 301,
     REG_S1 = 302,
     REG_S2 = 303,
     REG_L1 = 304,
     REG_L2 = 305,
     NUMERIC = 306,
     STRING = 307,
     STRING_ID = 308,
     TRUE_TOKEN = 309,
     FALSE_TOKEN = 310,
     I_LD = 311,
     I_LNIL = 312,
     NIL = 313,
     I_ADD = 314,
     I_SUB = 315,
     I_MUL = 316,
     I_DIV = 317,
     I_MOD = 318,
     I_POW = 319,
     I_ADDS = 320,
     I_SUBS = 321,
     I_MULS = 322,
     I_DIVS = 323,
     I_POWS = 324,
     I_INC = 325,
     I_DEC = 326,
     I_INCP = 327,
     I_DECP = 328,
     I_NEG = 329,
     I_NOT = 330,
     I_RET = 331,
     I_RETV = 332,
     I_RETA = 333,
     I_FORK = 334,
     I_PUSH = 335,
     I_PSHR = 336,
     I_PSHN = 337,
     I_POP = 338,
     I_LDV = 339,
     I_LDVT = 340,
     I_STV = 341,
     I_STVR = 342,
     I_STVS = 343,
     I_LDP = 344,
     I_LDPT = 345,
     I_STP = 346,
     I_STPR = 347,
     I_STPS = 348,
     I_TRAV = 349,
     I_TRAN = 350,
     I_TRAL = 351,
     I_IPOP = 352,
     I_XPOP = 353,
     I_GENA = 354,
     I_GEND = 355,
     I_GENR = 356,
     I_GEOR = 357,
     I_JMP = 358,
     I_IFT = 359,
     I_IFF = 360,
     I_BOOL = 361,
     I_EQ = 362,
     I_NEQ = 363,
     I_GT = 364,
     I_GE = 365,
     I_LT = 366,
     I_LE = 367,
     I_UNPK = 368,
     I_UNPS = 369,
     I_CALL = 370,
     I_INST = 371,
     I_SWCH = 372,
     I_SELE = 373,
     I_NOP = 374,
     I_TRY = 375,
     I_JTRY = 376,
     I_PTRY = 377,
     I_RIS = 378,
     I_LDRF = 379,
     I_ONCE = 380,
     I_BAND = 381,
     I_BOR = 382,
     I_BXOR = 383,
     I_BNOT = 384,
     I_MODS = 385,
     I_AND = 386,
     I_OR = 387,
     I_ANDS = 388,
     I_ORS = 389,
     I_XORS = 390,
     I_NOTS = 391,
     I_HAS = 392,
     I_HASN = 393,
     I_GIVE = 394,
     I_GIVN = 395,
     I_IN = 396,
     I_NOIN = 397,
     I_PROV = 398,
     I_END = 399,
     I_PEEK = 400,
     I_PSIN = 401,
     I_PASS = 402,
     I_SHR = 403,
     I_SHL = 404,
     I_SHRS = 405,
     I_SHLS = 406,
     I_LDVR = 407,
     I_LDPR = 408,
     I_LSB = 409,
     I_INDI = 410,
     I_STEX = 411,
     I_TRAC = 412,
     I_WRT = 413,
     I_STO = 414,
     I_FORB = 415
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
#define DIMPORT 283
#define DSWITCH 284
#define DSELECT 285
#define DCASE 286
#define DENDSWITCH 287
#define DLINE 288
#define DSTRING 289
#define DISTRING 290
#define DCSTRING 291
#define DHAS 292
#define DHASNT 293
#define DINHERIT 294
#define DINSTANCE 295
#define SYMBOL 296
#define EXPORT 297
#define LABEL 298
#define INTEGER 299
#define REG_A 300
#define REG_B 301
#define REG_S1 302
#define REG_S2 303
#define REG_L1 304
#define REG_L2 305
#define NUMERIC 306
#define STRING 307
#define STRING_ID 308
#define TRUE_TOKEN 309
#define FALSE_TOKEN 310
#define I_LD 311
#define I_LNIL 312
#define NIL 313
#define I_ADD 314
#define I_SUB 315
#define I_MUL 316
#define I_DIV 317
#define I_MOD 318
#define I_POW 319
#define I_ADDS 320
#define I_SUBS 321
#define I_MULS 322
#define I_DIVS 323
#define I_POWS 324
#define I_INC 325
#define I_DEC 326
#define I_INCP 327
#define I_DECP 328
#define I_NEG 329
#define I_NOT 330
#define I_RET 331
#define I_RETV 332
#define I_RETA 333
#define I_FORK 334
#define I_PUSH 335
#define I_PSHR 336
#define I_PSHN 337
#define I_POP 338
#define I_LDV 339
#define I_LDVT 340
#define I_STV 341
#define I_STVR 342
#define I_STVS 343
#define I_LDP 344
#define I_LDPT 345
#define I_STP 346
#define I_STPR 347
#define I_STPS 348
#define I_TRAV 349
#define I_TRAN 350
#define I_TRAL 351
#define I_IPOP 352
#define I_XPOP 353
#define I_GENA 354
#define I_GEND 355
#define I_GENR 356
#define I_GEOR 357
#define I_JMP 358
#define I_IFT 359
#define I_IFF 360
#define I_BOOL 361
#define I_EQ 362
#define I_NEQ 363
#define I_GT 364
#define I_GE 365
#define I_LT 366
#define I_LE 367
#define I_UNPK 368
#define I_UNPS 369
#define I_CALL 370
#define I_INST 371
#define I_SWCH 372
#define I_SELE 373
#define I_NOP 374
#define I_TRY 375
#define I_JTRY 376
#define I_PTRY 377
#define I_RIS 378
#define I_LDRF 379
#define I_ONCE 380
#define I_BAND 381
#define I_BOR 382
#define I_BXOR 383
#define I_BNOT 384
#define I_MODS 385
#define I_AND 386
#define I_OR 387
#define I_ANDS 388
#define I_ORS 389
#define I_XORS 390
#define I_NOTS 391
#define I_HAS 392
#define I_HASN 393
#define I_GIVE 394
#define I_GIVN 395
#define I_IN 396
#define I_NOIN 397
#define I_PROV 398
#define I_END 399
#define I_PEEK 400
#define I_PSIN 401
#define I_PASS 402
#define I_SHR 403
#define I_SHL 404
#define I_SHRS 405
#define I_SHLS 406
#define I_LDVR 407
#define I_LDPR 408
#define I_LSB 409
#define I_INDI 410
#define I_STEX 411
#define I_TRAC 412
#define I_WRT 413
#define I_STO 414
#define I_FORB 415




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif




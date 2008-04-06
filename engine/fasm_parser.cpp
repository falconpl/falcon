/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse fasm_parse
#define yylex   fasm_lex
#define yyerror fasm_error
#define yylval  fasm_lval
#define yychar  fasm_char
#define yydebug fasm_debug
#define yynerrs fasm_nerrs


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
     DCSTRING = 289,
     DHAS = 290,
     DHASNT = 291,
     DINHERIT = 292,
     DINSTANCE = 293,
     SYMBOL = 294,
     EXPORT = 295,
     LABEL = 296,
     INTEGER = 297,
     REG_A = 298,
     REG_B = 299,
     REG_S1 = 300,
     REG_S2 = 301,
     NUMERIC = 302,
     STRING = 303,
     STRING_ID = 304,
     TRUE_TOKEN = 305,
     FALSE_TOKEN = 306,
     I_LD = 307,
     I_LNIL = 308,
     NIL = 309,
     I_ADD = 310,
     I_SUB = 311,
     I_MUL = 312,
     I_DIV = 313,
     I_MOD = 314,
     I_POW = 315,
     I_ADDS = 316,
     I_SUBS = 317,
     I_MULS = 318,
     I_DIVS = 319,
     I_POWS = 320,
     I_INC = 321,
     I_DEC = 322,
     I_INCP = 323,
     I_DECP = 324,
     I_NEG = 325,
     I_NOT = 326,
     I_RET = 327,
     I_RETV = 328,
     I_RETA = 329,
     I_FORK = 330,
     I_PUSH = 331,
     I_PSHR = 332,
     I_PSHN = 333,
     I_POP = 334,
     I_LDV = 335,
     I_LDVT = 336,
     I_STV = 337,
     I_STVR = 338,
     I_STVS = 339,
     I_LDP = 340,
     I_LDPT = 341,
     I_STP = 342,
     I_STPR = 343,
     I_STPS = 344,
     I_TRAV = 345,
     I_TRAN = 346,
     I_TRAL = 347,
     I_IPOP = 348,
     I_XPOP = 349,
     I_GENA = 350,
     I_GEND = 351,
     I_GENR = 352,
     I_GEOR = 353,
     I_JMP = 354,
     I_IFT = 355,
     I_IFF = 356,
     I_BOOL = 357,
     I_EQ = 358,
     I_NEQ = 359,
     I_GT = 360,
     I_GE = 361,
     I_LT = 362,
     I_LE = 363,
     I_UNPK = 364,
     I_UNPS = 365,
     I_CALL = 366,
     I_INST = 367,
     I_SWCH = 368,
     I_SELE = 369,
     I_NOP = 370,
     I_TRY = 371,
     I_JTRY = 372,
     I_PTRY = 373,
     I_RIS = 374,
     I_LDRF = 375,
     I_ONCE = 376,
     I_BAND = 377,
     I_BOR = 378,
     I_BXOR = 379,
     I_BNOT = 380,
     I_MODS = 381,
     I_AND = 382,
     I_OR = 383,
     I_ANDS = 384,
     I_ORS = 385,
     I_XORS = 386,
     I_NOTS = 387,
     I_HAS = 388,
     I_HASN = 389,
     I_GIVE = 390,
     I_GIVN = 391,
     I_IN = 392,
     I_NOIN = 393,
     I_PROV = 394,
     I_END = 395,
     I_PEEK = 396,
     I_PSIN = 397,
     I_PASS = 398,
     I_SHR = 399,
     I_SHL = 400,
     I_SHRS = 401,
     I_SHLS = 402,
     I_LDVR = 403,
     I_LDPR = 404,
     I_LSB = 405,
     I_INDI = 406,
     I_STEX = 407,
     I_TRAC = 408,
     I_WRT = 409,
     I_STO = 410
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
#define DCSTRING 289
#define DHAS 290
#define DHASNT 291
#define DINHERIT 292
#define DINSTANCE 293
#define SYMBOL 294
#define EXPORT 295
#define LABEL 296
#define INTEGER 297
#define REG_A 298
#define REG_B 299
#define REG_S1 300
#define REG_S2 301
#define NUMERIC 302
#define STRING 303
#define STRING_ID 304
#define TRUE_TOKEN 305
#define FALSE_TOKEN 306
#define I_LD 307
#define I_LNIL 308
#define NIL 309
#define I_ADD 310
#define I_SUB 311
#define I_MUL 312
#define I_DIV 313
#define I_MOD 314
#define I_POW 315
#define I_ADDS 316
#define I_SUBS 317
#define I_MULS 318
#define I_DIVS 319
#define I_POWS 320
#define I_INC 321
#define I_DEC 322
#define I_INCP 323
#define I_DECP 324
#define I_NEG 325
#define I_NOT 326
#define I_RET 327
#define I_RETV 328
#define I_RETA 329
#define I_FORK 330
#define I_PUSH 331
#define I_PSHR 332
#define I_PSHN 333
#define I_POP 334
#define I_LDV 335
#define I_LDVT 336
#define I_STV 337
#define I_STVR 338
#define I_STVS 339
#define I_LDP 340
#define I_LDPT 341
#define I_STP 342
#define I_STPR 343
#define I_STPS 344
#define I_TRAV 345
#define I_TRAN 346
#define I_TRAL 347
#define I_IPOP 348
#define I_XPOP 349
#define I_GENA 350
#define I_GEND 351
#define I_GENR 352
#define I_GEOR 353
#define I_JMP 354
#define I_IFT 355
#define I_IFF 356
#define I_BOOL 357
#define I_EQ 358
#define I_NEQ 359
#define I_GT 360
#define I_GE 361
#define I_LT 362
#define I_LE 363
#define I_UNPK 364
#define I_UNPS 365
#define I_CALL 366
#define I_INST 367
#define I_SWCH 368
#define I_SELE 369
#define I_NOP 370
#define I_TRY 371
#define I_JTRY 372
#define I_PTRY 373
#define I_RIS 374
#define I_LDRF 375
#define I_ONCE 376
#define I_BAND 377
#define I_BOR 378
#define I_BXOR 379
#define I_BNOT 380
#define I_MODS 381
#define I_AND 382
#define I_OR 383
#define I_ANDS 384
#define I_ORS 385
#define I_XORS 386
#define I_NOTS 387
#define I_HAS 388
#define I_HASN 389
#define I_GIVE 390
#define I_GIVN 391
#define I_IN 392
#define I_NOIN 393
#define I_PROV 394
#define I_END 395
#define I_PEEK 396
#define I_PSIN 397
#define I_PASS 398
#define I_SHR 399
#define I_SHL 400
#define I_SHRS 401
#define I_SHLS 402
#define I_LDVR 403
#define I_LDPR 404
#define I_LSB 405
#define I_INDI 406
#define I_STEX 407
#define I_TRAC 408
#define I_WRT 409
#define I_STO 410




/* Copy the first part of user declarations.  */
#line 17 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"

#include <falcon/setup.h>
#include <stdio.h>
#include <iostream>
#include <ctype.h>

#include <fasm/comp.h>
#include <fasm/clexer.h>
#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/pcodes.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::AsmCompiler *>(fasm_param) )
#define  LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM fasm_param
#define YYLEX_PARAM fasm_param
#define YYSTYPE Falcon::Pseudo *

int fasm_parse( void *param );
void fasm_error( const char *s );

inline int yylex (void *lvalp, void *fasm_param)
{
   return COMPILER->lexer()->doLex( lvalp );
}

/* Cures a bug in bison 1.8  */
#undef __GNUC_MINOR__



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 462 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1546

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  156
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  123
/* YYNRULES -- Number of rules.  */
#define YYNRULES  438
/* YYNRULES -- Number of states.  */
#define YYNSTATES  794

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   410

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    12,    16,    19,    22,
      25,    27,    29,    31,    33,    35,    37,    39,    41,    43,
      45,    47,    49,    51,    53,    55,    57,    59,    61,    63,
      65,    67,    69,    72,    76,    81,    86,    92,    96,   101,
     105,   110,   114,   118,   121,   125,   129,   134,   136,   139,
     144,   149,   154,   159,   164,   169,   174,   179,   184,   191,
     193,   197,   201,   205,   208,   211,   216,   222,   226,   231,
     234,   238,   241,   243,   244,   249,   252,   256,   259,   262,
     265,   267,   271,   273,   277,   280,   281,   283,   287,   289,
     291,   292,   294,   296,   298,   300,   302,   304,   306,   308,
     310,   312,   314,   316,   318,   320,   322,   324,   326,   328,
     330,   332,   334,   336,   338,   340,   342,   344,   346,   348,
     350,   352,   354,   356,   358,   360,   362,   364,   366,   368,
     370,   372,   374,   376,   378,   380,   382,   384,   386,   388,
     390,   392,   394,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   414,   416,   418,   420,   422,   424,   426,   428,
     430,   432,   434,   436,   438,   440,   442,   444,   446,   448,
     450,   452,   454,   456,   458,   460,   462,   464,   466,   468,
     470,   472,   474,   476,   478,   480,   482,   484,   486,   488,
     490,   492,   494,   496,   498,   503,   506,   511,   514,   517,
     520,   525,   528,   533,   536,   541,   544,   549,   552,   557,
     560,   565,   568,   573,   576,   581,   584,   589,   592,   597,
     600,   605,   608,   613,   616,   621,   624,   629,   632,   637,
     640,   645,   648,   651,   654,   657,   660,   663,   666,   669,
     672,   675,   678,   681,   684,   687,   690,   693,   698,   703,
     706,   711,   716,   719,   724,   727,   732,   735,   738,   740,
     743,   746,   749,   752,   755,   758,   761,   764,   767,   772,
     777,   780,   787,   794,   797,   804,   811,   814,   821,   828,
     831,   836,   841,   844,   849,   854,   857,   864,   871,   874,
     881,   888,   891,   898,   905,   908,   913,   918,   921,   928,
     935,   938,   945,   952,   955,   958,   961,   964,   967,   970,
     973,   976,   979,   982,   989,   992,   995,   998,  1001,  1004,
    1007,  1010,  1013,  1016,  1019,  1024,  1029,  1032,  1037,  1042,
    1045,  1050,  1055,  1058,  1061,  1064,  1067,  1069,  1072,  1074,
    1077,  1080,  1083,  1085,  1088,  1091,  1094,  1096,  1099,  1106,
    1109,  1116,  1119,  1123,  1129,  1134,  1139,  1142,  1147,  1152,
    1157,  1162,  1165,  1170,  1175,  1180,  1185,  1188,  1193,  1198,
    1203,  1208,  1211,  1214,  1217,  1220,  1225,  1228,  1233,  1236,
    1241,  1244,  1249,  1252,  1257,  1260,  1265,  1268,  1273,  1276,
    1279,  1282,  1287,  1292,  1295,  1300,  1305,  1308,  1313,  1318,
    1321,  1326,  1331,  1334,  1339,  1342,  1347,  1350,  1355,  1358,
    1361,  1364,  1367,  1370,  1375,  1378,  1383,  1386,  1391,  1394,
    1399,  1402,  1407,  1410,  1415,  1418,  1423,  1426,  1429,  1432,
    1435,  1438,  1441,  1444,  1447,  1450,  1453,  1456,  1461
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     157,     0,    -1,    -1,   157,   158,    -1,     3,    -1,   167,
       3,    -1,   171,   175,     3,    -1,   171,     3,    -1,   175,
       3,    -1,     1,     3,    -1,    54,    -1,   160,    -1,   161,
      -1,   164,    -1,    39,    -1,   162,    -1,    43,    -1,    44,
      -1,    45,    -1,    46,    -1,    54,    -1,   164,    -1,    47,
      -1,    50,    -1,    51,    -1,   165,    -1,    49,    -1,    48,
      -1,    42,    -1,    49,    -1,    48,    -1,     7,    -1,    26,
       4,    -1,     8,     4,   174,    -1,     8,     4,   174,    40,
      -1,     9,     4,   163,   174,    -1,     9,     4,   163,   174,
      40,    -1,    10,     4,   163,    -1,    10,     4,   163,    40,
      -1,    11,     4,   174,    -1,    11,     4,   174,    40,    -1,
      12,     4,   174,    -1,    13,     4,   174,    -1,    14,     4,
      -1,    14,     4,    40,    -1,    16,     4,   174,    -1,    16,
       4,   174,    40,    -1,    15,    -1,    27,     4,    -1,    28,
     161,     5,     4,    -1,    28,   161,     5,    42,    -1,    29,
     161,     5,     4,    -1,    29,   161,     5,    42,    -1,    30,
      54,     5,     4,    -1,    30,    42,     5,     4,    -1,    30,
      48,     5,     4,    -1,    30,    49,     5,     4,    -1,    30,
      39,     5,     4,    -1,    30,    42,     6,    42,     5,     4,
      -1,    31,    -1,    18,     4,   163,    -1,    18,     4,    39,
      -1,    19,     4,    39,    -1,    35,   169,    -1,    36,   170,
      -1,    38,    39,     4,   174,    -1,    38,    39,     4,   174,
      40,    -1,    20,     4,   174,    -1,    20,     4,   174,    40,
      -1,    21,     4,    -1,    21,     4,    40,    -1,    22,    39,
      -1,    23,    -1,    -1,    37,    39,   168,   172,    -1,    24,
       4,    -1,    25,     4,   174,    -1,    32,    42,    -1,    33,
      48,    -1,    34,    48,    -1,    39,    -1,   169,     5,    39,
      -1,    39,    -1,   170,     5,    39,    -1,    41,     6,    -1,
      -1,   173,    -1,   172,     5,   173,    -1,   163,    -1,    39,
      -1,    -1,    42,    -1,   176,    -1,   178,    -1,   179,    -1,
     181,    -1,   183,    -1,   185,    -1,   187,    -1,   188,    -1,
     180,    -1,   182,    -1,   184,    -1,   186,    -1,   196,    -1,
     197,    -1,   198,    -1,   199,    -1,   189,    -1,   190,    -1,
     191,    -1,   192,    -1,   193,    -1,   194,    -1,   200,    -1,
     201,    -1,   206,    -1,   207,    -1,   208,    -1,   210,    -1,
     211,    -1,   212,    -1,   213,    -1,   214,    -1,   215,    -1,
     216,    -1,   217,    -1,   218,    -1,   220,    -1,   219,    -1,
     221,    -1,   222,    -1,   223,    -1,   224,    -1,   225,    -1,
     226,    -1,   227,    -1,   228,    -1,   230,    -1,   231,    -1,
     232,    -1,   233,    -1,   234,    -1,   204,    -1,   205,    -1,
     202,    -1,   203,    -1,   236,    -1,   238,    -1,   237,    -1,
     239,    -1,   195,    -1,   240,    -1,   235,    -1,   229,    -1,
     242,    -1,   243,    -1,   177,    -1,   241,    -1,   246,    -1,
     247,    -1,   248,    -1,   249,    -1,   250,    -1,   251,    -1,
     252,    -1,   253,    -1,   254,    -1,   257,    -1,   255,    -1,
     256,    -1,   258,    -1,   259,    -1,   260,    -1,   261,    -1,
     262,    -1,   263,    -1,   264,    -1,   245,    -1,   209,    -1,
     265,    -1,   266,    -1,   267,    -1,   268,    -1,   269,    -1,
     270,    -1,   271,    -1,   272,    -1,   273,    -1,   274,    -1,
     275,    -1,   276,    -1,   277,    -1,   278,    -1,    52,   161,
       5,   159,    -1,    52,     1,    -1,   120,   161,     5,   159,
      -1,   120,     1,    -1,    53,   161,    -1,    53,     1,    -1,
      55,   159,     5,   159,    -1,    55,     1,    -1,    61,   161,
       5,   159,    -1,    61,     1,    -1,    56,   159,     5,   159,
      -1,    56,     1,    -1,    62,   161,     5,   159,    -1,    62,
       1,    -1,    57,   159,     5,   159,    -1,    57,     1,    -1,
      63,   161,     5,   159,    -1,    63,     1,    -1,    58,   159,
       5,   159,    -1,    58,     1,    -1,    64,   161,     5,   159,
      -1,    64,     1,    -1,    59,   159,     5,   159,    -1,    59,
       1,    -1,    60,   159,     5,   159,    -1,    60,     1,    -1,
     103,   159,     5,   159,    -1,   103,     1,    -1,   104,   159,
       5,   159,    -1,   104,     1,    -1,   106,   159,     5,   159,
      -1,   106,     1,    -1,   105,   159,     5,   159,    -1,   105,
       1,    -1,   108,   159,     5,   159,    -1,   108,     1,    -1,
     107,   159,     5,   159,    -1,   107,     1,    -1,   116,     4,
      -1,   116,    42,    -1,   116,     1,    -1,    66,   161,    -1,
      66,     1,    -1,    67,   161,    -1,    67,     1,    -1,    68,
     161,    -1,    68,     1,    -1,    69,   161,    -1,    69,     1,
      -1,    70,   159,    -1,    70,     1,    -1,    71,   159,    -1,
      71,     1,    -1,   111,    42,     5,   161,    -1,   111,    42,
       5,     4,    -1,   111,     1,    -1,   112,    42,     5,   161,
      -1,   112,    42,     5,     4,    -1,   112,     1,    -1,   109,
     161,     5,   161,    -1,   109,     1,    -1,   110,    42,     5,
     161,    -1,   110,     1,    -1,    76,   159,    -1,    78,    -1,
      76,     1,    -1,    77,   159,    -1,    77,     1,    -1,    79,
     161,    -1,    79,     1,    -1,   141,   161,    -1,   141,     1,
      -1,    94,   161,    -1,    94,     1,    -1,    80,   161,     5,
     161,    -1,    80,   161,     5,   165,    -1,    80,     1,    -1,
      81,   161,     5,   161,     5,   161,    -1,    81,   161,     5,
     165,     5,   161,    -1,    81,     1,    -1,    82,   161,     5,
     161,     5,   159,    -1,    82,   161,     5,   165,     5,   159,
      -1,    82,     1,    -1,    83,   161,     5,   161,     5,   159,
      -1,    83,   161,     5,   165,     5,   159,    -1,    83,     1,
      -1,    84,   161,     5,   161,    -1,    84,   161,     5,   165,
      -1,    84,     1,    -1,    85,   161,     5,   161,    -1,    85,
     161,     5,   165,    -1,    85,     1,    -1,    86,   161,     5,
     161,     5,   161,    -1,    86,   161,     5,   165,     5,   161,
      -1,    86,     1,    -1,    87,   161,     5,   161,     5,   159,
      -1,    87,   161,     5,   165,     5,   159,    -1,    87,     1,
      -1,    88,   161,     5,   161,     5,   161,    -1,    88,   161,
       5,   165,     5,   161,    -1,    88,     1,    -1,    89,   161,
       5,   161,    -1,    89,   161,     5,   165,    -1,    89,     1,
      -1,    90,    42,     5,   159,     5,   159,    -1,    90,     4,
       5,   159,     5,   159,    -1,    90,     1,    -1,    91,     4,
       5,     4,     5,    42,    -1,    91,    42,     5,    42,     5,
      42,    -1,    91,     1,    -1,    92,    42,    -1,    92,     4,
      -1,    92,     1,    -1,    93,    42,    -1,    93,     1,    -1,
      95,    42,    -1,    95,     1,    -1,    96,    42,    -1,    96,
       1,    -1,    97,   160,     5,   160,     5,   163,    -1,    97,
       1,    -1,    98,   160,    -1,    98,     1,    -1,   119,   159,
      -1,   119,     1,    -1,    99,     4,    -1,    99,    42,    -1,
      99,     1,    -1,   102,   159,    -1,   102,     1,    -1,   100,
       4,     5,   159,    -1,   100,    42,     5,   159,    -1,   100,
       1,    -1,   101,     4,     5,   159,    -1,   101,    42,     5,
     159,    -1,   101,     1,    -1,    75,    42,     5,     4,    -1,
      75,    42,     5,    42,    -1,    75,     1,    -1,   117,     4,
      -1,   117,    42,    -1,   117,     1,    -1,    72,    -1,    72,
       1,    -1,    74,    -1,    74,     1,    -1,    73,   159,    -1,
      73,     1,    -1,   115,    -1,   115,     1,    -1,   118,    42,
      -1,   118,     1,    -1,   140,    -1,   140,     1,    -1,   113,
      42,     5,   161,     5,   244,    -1,   113,     1,    -1,   114,
      42,     5,   161,     5,   244,    -1,   114,     1,    -1,    42,
       5,     4,    -1,   244,     5,    42,     5,     4,    -1,   121,
      42,     5,   159,    -1,   121,     4,     5,   159,    -1,   121,
       1,    -1,   122,    39,     5,    39,    -1,   122,    39,     5,
      42,    -1,   122,    42,     5,    39,    -1,   122,    42,     5,
      42,    -1,   122,     1,    -1,   123,    39,     5,    39,    -1,
     123,    39,     5,    42,    -1,   123,    42,     5,    39,    -1,
     123,    42,     5,    42,    -1,   123,     1,    -1,   124,    39,
       5,    39,    -1,   124,    39,     5,    42,    -1,   124,    42,
       5,    39,    -1,   124,    42,     5,    42,    -1,   124,     1,
      -1,   125,    39,    -1,   125,    42,    -1,   125,     1,    -1,
     127,   159,     5,   159,    -1,   127,     1,    -1,   128,   159,
       5,   159,    -1,   128,     1,    -1,   129,   161,     5,   160,
      -1,   129,     1,    -1,   130,   161,     5,   160,    -1,   130,
       1,    -1,   131,   161,     5,   160,    -1,   131,     1,    -1,
     126,   161,     5,   160,    -1,   126,     1,    -1,    65,   161,
       5,   160,    -1,    65,     1,    -1,   132,   161,    -1,   132,
       1,    -1,   133,   161,     5,   161,    -1,   133,   161,     5,
      42,    -1,   133,     1,    -1,   134,   161,     5,   161,    -1,
     134,   161,     5,    42,    -1,   134,     1,    -1,   135,   161,
       5,   161,    -1,   135,   161,     5,    42,    -1,   135,     1,
      -1,   136,   161,     5,   161,    -1,   136,   161,     5,    42,
      -1,   136,     1,    -1,   137,   159,     5,   160,    -1,   137,
       1,    -1,   138,   159,     5,   160,    -1,   138,     1,    -1,
     139,   159,     5,   160,    -1,   139,     1,    -1,   142,   161,
      -1,   142,     1,    -1,   143,   161,    -1,   143,     1,    -1,
     144,   159,     5,   159,    -1,   144,     1,    -1,   145,   159,
       5,   159,    -1,   145,     1,    -1,   146,   159,     5,   159,
      -1,   146,     1,    -1,   147,   159,     5,   159,    -1,   147,
       1,    -1,   148,   159,     5,   159,    -1,   148,     1,    -1,
     149,   159,     5,   159,    -1,   149,     1,    -1,   150,   159,
       5,   159,    -1,   150,     1,    -1,   151,   166,    -1,   151,
     161,    -1,   151,     1,    -1,   152,   166,    -1,   152,   161,
      -1,   152,     1,    -1,   153,   159,    -1,   153,     1,    -1,
     154,   159,    -1,   154,     1,    -1,   155,   161,     5,   159,
      -1,   155,     1,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   226,   226,   228,   232,   233,   234,   235,   236,   237,
     240,   240,   241,   241,   242,   242,   243,   243,   243,   243,
     245,   245,   246,   246,   246,   246,   247,   247,   247,   248,
     248,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
     280,   281,   282,   283,   284,   285,   286,   287,   288,   289,
     290,   291,   292,   293,   293,   294,   295,   296,   297,   302,
     311,   312,   316,   317,   320,   322,   324,   325,   329,   330,
     334,   335,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   350,   351,   352,   353,   354,   355,   356,
     357,   358,   359,   360,   361,   362,   363,   364,   365,   366,
     367,   368,   369,   370,   371,   372,   373,   374,   375,   376,
     377,   378,   379,   380,   381,   382,   383,   384,   385,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,   406,
     407,   408,   409,   410,   411,   412,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   444,   445,   449,   450,   454,   455,
     459,   460,   464,   465,   470,   471,   475,   476,   480,   481,
     485,   486,   491,   492,   496,   497,   501,   502,   506,   507,
     512,   513,   517,   518,   522,   523,   527,   528,   532,   533,
     537,   538,   542,   543,   544,   548,   549,   553,   554,   559,
     560,   564,   565,   570,   571,   575,   576,   580,   581,   582,
     586,   587,   588,   592,   593,   597,   598,   603,   604,   605,
     609,   610,   615,   616,   620,   621,   625,   626,   631,   632,
     633,   637,   638,   639,   643,   644,   645,   649,   650,   651,
     655,   656,   657,   661,   662,   663,   667,   668,   669,   673,
     674,   675,   679,   680,   681,   685,   686,   687,   691,   692,
     693,   697,   698,   699,   703,   704,   705,   709,   710,   714,
     715,   719,   720,   724,   725,   729,   730,   734,   735,   739,
     740,   741,   745,   746,   750,   751,   752,   756,   757,   758,
     763,   764,   765,   769,   770,   771,   775,   776,   780,   781,
     785,   786,   790,   791,   795,   796,   800,   801,   805,   806,
     810,   811,   815,   824,   833,   834,   835,   839,   840,   841,
     842,   843,   847,   848,   849,   850,   851,   855,   856,   857,
     858,   859,   863,   864,   865,   869,   870,   874,   875,   879,
     880,   884,   885,   889,   890,   894,   895,   899,   900,   904,
     905,   909,   910,   911,   915,   916,   917,   921,   922,   923,
     927,   928,   929,   934,   935,   939,   940,   944,   945,   949,
     950,   954,   955,   959,   960,   964,   965,   969,   970,   974,
     975,   979,   980,   984,   985,   989,   990,   994,   995,   996,
    1000,  1001,  1002,  1006,  1007,  1011,  1012,  1017,  1018
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "NAME", "COMMA", "COLON",
  "DENTRY", "DGLOBAL", "DVAR", "DCONST", "DATTRIB", "DLOCAL", "DPARAM",
  "DFUNCDEF", "DENDFUNC", "DFUNC", "DMETHOD", "DPROP", "DPROPREF",
  "DCLASS", "DCLASSDEF", "DCTOR", "DENDCLASS", "DFROM", "DEXTERN",
  "DMODULE", "DLOAD", "DSWITCH", "DSELECT", "DCASE", "DENDSWITCH", "DLINE",
  "DSTRING", "DCSTRING", "DHAS", "DHASNT", "DINHERIT", "DINSTANCE",
  "SYMBOL", "EXPORT", "LABEL", "INTEGER", "REG_A", "REG_B", "REG_S1",
  "REG_S2", "NUMERIC", "STRING", "STRING_ID", "TRUE_TOKEN", "FALSE_TOKEN",
  "I_LD", "I_LNIL", "NIL", "I_ADD", "I_SUB", "I_MUL", "I_DIV", "I_MOD",
  "I_POW", "I_ADDS", "I_SUBS", "I_MULS", "I_DIVS", "I_POWS", "I_INC",
  "I_DEC", "I_INCP", "I_DECP", "I_NEG", "I_NOT", "I_RET", "I_RETV",
  "I_RETA", "I_FORK", "I_PUSH", "I_PSHR", "I_PSHN", "I_POP", "I_LDV",
  "I_LDVT", "I_STV", "I_STVR", "I_STVS", "I_LDP", "I_LDPT", "I_STP",
  "I_STPR", "I_STPS", "I_TRAV", "I_TRAN", "I_TRAL", "I_IPOP", "I_XPOP",
  "I_GENA", "I_GEND", "I_GENR", "I_GEOR", "I_JMP", "I_IFT", "I_IFF",
  "I_BOOL", "I_EQ", "I_NEQ", "I_GT", "I_GE", "I_LT", "I_LE", "I_UNPK",
  "I_UNPS", "I_CALL", "I_INST", "I_SWCH", "I_SELE", "I_NOP", "I_TRY",
  "I_JTRY", "I_PTRY", "I_RIS", "I_LDRF", "I_ONCE", "I_BAND", "I_BOR",
  "I_BXOR", "I_BNOT", "I_MODS", "I_AND", "I_OR", "I_ANDS", "I_ORS",
  "I_XORS", "I_NOTS", "I_HAS", "I_HASN", "I_GIVE", "I_GIVN", "I_IN",
  "I_NOIN", "I_PROV", "I_END", "I_PEEK", "I_PSIN", "I_PASS", "I_SHR",
  "I_SHL", "I_SHRS", "I_SHLS", "I_LDVR", "I_LDPR", "I_LSB", "I_INDI",
  "I_STEX", "I_TRAC", "I_WRT", "I_STO", "$accept", "input", "line",
  "xoperand", "operand", "op_variable", "op_register", "x_op_immediate",
  "op_immediate", "op_scalar", "op_string", "directive", "@1",
  "has_symlist", "hasnt_symlist", "label", "inherit_param_list",
  "inherit_param", "def_line", "instruction", "inst_ld", "inst_ldrf",
  "inst_ldnil", "inst_add", "inst_adds", "inst_sub", "inst_subs",
  "inst_mul", "inst_muls", "inst_div", "inst_divs", "inst_mod", "inst_pow",
  "inst_eq", "inst_ne", "inst_ge", "inst_gt", "inst_le", "inst_lt",
  "inst_try", "inst_inc", "inst_dec", "inst_incp", "inst_decp", "inst_neg",
  "inst_not", "inst_call", "inst_inst", "inst_unpk", "inst_unps",
  "inst_push", "inst_pshr", "inst_pop", "inst_peek", "inst_xpop",
  "inst_ldv", "inst_ldvt", "inst_stv", "inst_stvr", "inst_stvs",
  "inst_ldp", "inst_ldpt", "inst_stp", "inst_stpr", "inst_stps",
  "inst_trav", "inst_tran", "inst_tral", "inst_ipop", "inst_gena",
  "inst_gend", "inst_genr", "inst_geor", "inst_ris", "inst_jmp",
  "inst_bool", "inst_ift", "inst_iff", "inst_fork", "inst_jtry",
  "inst_ret", "inst_reta", "inst_retval", "inst_nop", "inst_ptry",
  "inst_end", "inst_swch", "inst_sele", "switch_list", "inst_once",
  "inst_band", "inst_bor", "inst_bxor", "inst_bnot", "inst_and", "inst_or",
  "inst_ands", "inst_ors", "inst_xors", "inst_mods", "inst_pows",
  "inst_nots", "inst_has", "inst_hasn", "inst_give", "inst_givn",
  "inst_in", "inst_noin", "inst_prov", "inst_psin", "inst_pass",
  "inst_shr", "inst_shl", "inst_shrs", "inst_shls", "inst_ldvr",
  "inst_ldpr", "inst_lsb", "inst_indi", "inst_stex", "inst_trac",
  "inst_wrt", "inst_sto", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   156,   157,   157,   158,   158,   158,   158,   158,   158,
     159,   159,   160,   160,   161,   161,   162,   162,   162,   162,
     163,   163,   164,   164,   164,   164,   165,   165,   165,   166,
     166,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   168,   167,   167,   167,   167,   167,   167,
     169,   169,   170,   170,   171,   172,   172,   172,   173,   173,
     174,   174,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   176,   176,   177,   177,   178,   178,
     179,   179,   180,   180,   181,   181,   182,   182,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   188,   188,
     189,   189,   190,   190,   191,   191,   192,   192,   193,   193,
     194,   194,   195,   195,   195,   196,   196,   197,   197,   198,
     198,   199,   199,   200,   200,   201,   201,   202,   202,   202,
     203,   203,   203,   204,   204,   205,   205,   206,   206,   206,
     207,   207,   208,   208,   209,   209,   210,   210,   211,   211,
     211,   212,   212,   212,   213,   213,   213,   214,   214,   214,
     215,   215,   215,   216,   216,   216,   217,   217,   217,   218,
     218,   218,   219,   219,   219,   220,   220,   220,   221,   221,
     221,   222,   222,   222,   223,   223,   223,   224,   224,   225,
     225,   226,   226,   227,   227,   228,   228,   229,   229,   230,
     230,   230,   231,   231,   232,   232,   232,   233,   233,   233,
     234,   234,   234,   235,   235,   235,   236,   236,   237,   237,
     238,   238,   239,   239,   240,   240,   241,   241,   242,   242,
     243,   243,   244,   244,   245,   245,   245,   246,   246,   246,
     246,   246,   247,   247,   247,   247,   247,   248,   248,   248,
     248,   248,   249,   249,   249,   250,   250,   251,   251,   252,
     252,   253,   253,   254,   254,   255,   255,   256,   256,   257,
     257,   258,   258,   258,   259,   259,   259,   260,   260,   260,
     261,   261,   261,   262,   262,   263,   263,   264,   264,   265,
     265,   266,   266,   267,   267,   268,   268,   269,   269,   270,
     270,   271,   271,   272,   272,   273,   273,   274,   274,   274,
     275,   275,   275,   276,   276,   277,   277,   278,   278
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     2,     3,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     3,     4,     4,     5,     3,     4,     3,
       4,     3,     3,     2,     3,     3,     4,     1,     2,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     6,     1,
       3,     3,     3,     2,     2,     4,     5,     3,     4,     2,
       3,     2,     1,     0,     4,     2,     3,     2,     2,     2,
       1,     3,     1,     3,     2,     0,     1,     3,     1,     1,
       0,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     4,     2,     4,     2,     2,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     4,     2,     4,     2,     4,     2,     4,     2,
       4,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     4,     2,
       4,     4,     2,     4,     2,     4,     2,     2,     1,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     4,
       2,     6,     6,     2,     6,     6,     2,     6,     6,     2,
       4,     4,     2,     4,     4,     2,     6,     6,     2,     6,
       6,     2,     6,     6,     2,     4,     4,     2,     6,     6,
       2,     6,     6,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     6,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     2,     2,     2,     1,     2,     1,     2,
       2,     2,     1,     2,     2,     2,     1,     2,     6,     2,
       6,     2,     3,     5,     4,     4,     2,     4,     4,     4,
       4,     2,     4,     4,     4,     4,     2,     4,     4,     4,
       4,     2,     2,     2,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     4,     4,     2,     4,     4,     2,     4,     4,     2,
       4,     4,     2,     4,     2,     4,     2,     4,     2,     2,
       2,     2,     2,     4,     2,     4,     2,     4,     2,     4,
       2,     4,     2,     4,     2,     4,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     4,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,     4,    31,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,     0,     0,     0,
      72,     0,     0,     0,     0,     0,     0,     0,    59,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   258,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       3,     0,     0,     0,    92,   157,    93,    94,   100,    95,
     101,    96,   102,    97,   103,    98,    99,   108,   109,   110,
     111,   112,   113,   151,   104,   105,   106,   107,   114,   115,
     145,   146,   143,   144,   116,   117,   118,   179,   119,   120,
     121,   122,   123,   124,   125,   126,   127,   129,   128,   130,
     131,   132,   133,   134,   135,   136,   137,   154,   138,   139,
     140,   141,   142,   153,   147,   149,   148,   150,   152,   158,
     155,   156,   178,   159,   160,   161,   162,   163,   164,   165,
     166,   167,   169,   170,   168,   171,   172,   173,   174,   175,
     176,   177,   180,   181,   182,   183,   184,   185,   186,   187,
     188,   189,   190,   191,   192,   193,     9,    90,     0,     0,
      90,    90,    90,    43,    90,     0,     0,    90,    69,    71,
      75,    90,    32,    48,    14,    16,    17,    18,    19,     0,
      15,     0,     0,     0,     0,     0,     0,    77,    78,    79,
      80,    63,    82,    64,    73,     0,    84,   195,     0,   199,
     198,   201,    28,    22,    27,    26,    23,    24,    10,     0,
      11,    12,    13,    25,   205,     0,   209,     0,   213,     0,
     217,     0,   219,     0,   203,     0,   207,     0,   211,     0,
     215,     0,   388,     0,   236,   235,   238,   237,   240,   239,
     242,   241,   244,   243,   246,   245,   337,   341,   340,   339,
     332,     0,   259,   257,   261,   260,   263,   262,   270,     0,
     273,     0,   276,     0,   279,     0,   282,     0,   285,     0,
     288,     0,   291,     0,   294,     0,   297,     0,   300,     0,
       0,   303,     0,     0,   306,   305,   304,   308,   307,   267,
     266,   310,   309,   312,   311,   314,     0,   316,   315,   321,
     319,   320,   326,     0,     0,   329,     0,     0,   323,   322,
     221,     0,   223,     0,   227,     0,   225,     0,   231,     0,
     229,     0,   254,     0,   256,     0,   249,     0,   252,     0,
     349,     0,   351,     0,   343,   234,   232,   233,   335,   333,
     334,   345,   344,   318,   317,   197,     0,   356,     0,     0,
     361,     0,     0,   366,     0,     0,   371,     0,     0,   374,
     372,   373,   386,     0,   376,     0,   378,     0,   380,     0,
     382,     0,   384,     0,   390,   389,   393,     0,   396,     0,
     399,     0,   402,     0,   404,     0,   406,     0,   408,     0,
     347,   265,   264,   410,   409,   412,   411,   414,     0,   416,
       0,   418,     0,   420,     0,   422,     0,   424,     0,   426,
       0,   429,    30,    29,   428,   427,   432,   431,   430,   434,
     433,   436,   435,   438,     0,     5,     7,     0,     8,    91,
      33,    20,    90,    21,    37,    39,    41,    42,    44,    45,
      61,    60,    62,    67,    70,    76,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    85,    90,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     6,    34,    35,    38,    40,    46,    68,    49,
      50,    51,    52,    57,    54,     0,    55,    56,    53,    81,
      83,    89,    88,    74,    86,    65,   194,   200,   204,   208,
     212,   216,   218,   202,   206,   210,   214,   387,   330,   331,
     268,   269,     0,     0,     0,     0,     0,     0,   280,   281,
     283,   284,     0,     0,     0,     0,     0,     0,   295,   296,
       0,     0,     0,     0,     0,   324,   325,   327,   328,   220,
     222,   226,   224,   230,   228,   253,   255,   248,   247,   251,
     250,     0,     0,   196,   355,   354,   357,   358,   359,   360,
     362,   363,   364,   365,   367,   368,   369,   370,   385,   375,
     377,   379,   381,   383,   392,   391,   395,   394,   398,   397,
     401,   400,   403,   405,   407,   413,   415,   417,   419,   421,
     423,   425,   437,    36,     0,     0,    66,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    58,    87,   271,   272,
     274,   275,   277,   278,   286,   287,   289,   290,   292,   293,
     299,   298,   301,   302,   313,     0,   348,   350,     0,     0,
     352,     0,     0,   353
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,   140,   299,   300,   301,   270,   642,   302,   303,
     505,   141,   546,   281,   283,   142,   643,   644,   520,   143,
     144,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   155,   156,   157,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   170,   171,   172,   173,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,   197,   198,   199,   200,   201,   202,   203,
     204,   205,   206,   207,   208,   209,   210,   211,   786,   212,
     213,   214,   215,   216,   217,   218,   219,   220,   221,   222,
     223,   224,   225,   226,   227,   228,   229,   230,   231,   232,
     233,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -245
static const yytype_int16 yypact[] =
{
    -245,   305,  -245,     3,  -245,  -245,    12,    32,    60,    78,
      82,    94,   109,  -245,   112,   132,   158,   169,   178,   148,
    -245,   184,   212,   260,   277,   615,   615,   682,  -245,   254,
     263,   300,   314,   317,   320,   434,   470,   511,   586,    83,
     160,   176,   189,   211,   228,   625,   674,   726,   765,  1161,
    1174,  1187,  1200,  1208,   241,   460,   162,   547,   177,    18,
     648,   661,  -245,  1220,  1228,  1236,  1247,  1255,  1267,  1275,
    1283,  1294,  1302,  1314,    13,    33,    77,   114,  1322,   147,
     171,    96,   528,    79,   156,   175,   700,   713,   734,   752,
     927,   940,   953,  1330,   251,   267,   268,   302,   486,   180,
     262,   471,   488,   966,  1341,   473,    17,    57,   478,   482,
    1349,   979,   992,  1361,  1369,  1377,  1388,  1396,  1408,  1416,
    1424,  1005,  1018,  1031,   248,  1435,  1443,  1455,  1044,  1057,
    1070,  1083,  1096,  1109,  1122,    28,   306,  1135,  1148,  1463,
    -245,   475,   772,   481,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,   439,   888,   888,
     439,   439,   439,   446,   439,   443,   449,   439,   455,  -245,
    -245,   439,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   493,
    -245,   495,   513,   153,   526,   530,   541,  -245,  -245,  -245,
    -245,   542,  -245,   544,  -245,   512,  -245,  -245,   546,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   548,
    -245,  -245,  -245,  -245,  -245,   553,  -245,   554,  -245,   555,
    -245,   575,  -245,   576,  -245,   583,  -245,   594,  -245,   595,
    -245,   597,  -245,   599,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,   600,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   601,
    -245,   617,  -245,   618,  -245,   619,  -245,   622,  -245,   623,
    -245,   628,  -245,   631,  -245,   632,  -245,   633,  -245,   645,
     646,  -245,   647,   658,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,   662,  -245,  -245,  -245,
    -245,  -245,  -245,   667,   668,  -245,   669,   671,  -245,  -245,
    -245,   680,  -245,   681,  -245,   683,  -245,   684,  -245,   711,
    -245,   720,  -245,   721,  -245,   729,  -245,   732,  -245,   733,
    -245,   735,  -245,   736,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,   763,  -245,   769,   781,
    -245,   782,   784,  -245,   785,   787,  -245,   788,   800,  -245,
    -245,  -245,  -245,   802,  -245,   811,  -245,   812,  -245,   815,
    -245,   816,  -245,   821,  -245,  -245,  -245,   924,  -245,   926,
    -245,   928,  -245,   929,  -245,   952,  -245,   954,  -245,   955,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  1198,  -245,
    1203,  -245,  1205,  -245,  1206,  -245,  1207,  -245,  1209,  -245,
    1210,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1211,  -245,  -245,   549,  -245,  -245,
     529,  -245,   439,  -245,   613,   892,  -245,  -245,  -245,   918,
    -245,  -245,  -245,  1182,  -245,  -245,    27,    45,  1219,  1221,
    1185,  1224,  1230,  1231,  1197,  1199,   914,   439,   901,   901,
     901,   901,   901,   901,   901,   901,   901,   901,   901,  1479,
      66,  1492,  1492,  1492,  1492,  1492,  1492,  1492,  1492,  1492,
    1492,   901,   901,  1237,  1213,  1479,   901,   901,   901,   901,
     901,   901,   901,   901,   901,   901,   615,   615,   564,  1471,
     615,   615,   901,   901,   901,   -32,    -9,    -7,    23,    24,
      70,  1479,   901,   901,  1479,  1479,  1479,   125,   202,   255,
    1500,  1479,  1479,  1479,   901,   901,   901,   901,   901,   901,
     901,   901,  -245,  -245,  1184,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  1235,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  1244,  -245,  1202,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  1245,  1252,  1253,  1256,  1257,  1264,  -245,  -245,
    -245,  -245,  1265,  1272,  1273,  1278,  1280,  1282,  -245,  -245,
    1284,  1291,  1292,  1297,  1299,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  1300,  1303,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  1305,   914,  -245,   615,   615,   901,
     901,   901,   901,   615,   615,   901,   901,   615,   615,   901,
     901,  1218,  1246,   888,  1274,  1274,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  1312,  1319,  1319,  1321,  1288,
    -245,  1327,  1331,  -245
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -245,  -245,  -245,    63,   -79,   -25,  -245,  -240,  -244,   -99,
    1171,  -245,  -245,  -245,  -245,  -245,  -245,   589,  -200,  1194,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,   578,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,  -245,
    -245,  -245,  -245
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -347
static const yytype_int16 yytable[] =
{
     269,   271,   386,   388,   523,   523,   246,   706,   522,   524,
     707,   523,   288,   290,   368,   531,   247,   369,   440,   340,
     315,   317,   319,   321,   323,   325,   327,   329,   331,   501,
     708,   629,   710,   709,   371,   711,   248,   372,   347,   349,
     351,   353,   355,   357,   359,   361,   363,   365,   367,   631,
     525,   526,   527,   380,   529,   370,   441,   533,   443,   442,
     341,   535,   712,   714,   249,   713,   715,   264,   413,   630,
     658,   265,   266,   267,   268,   373,   502,   503,   374,   436,
     389,   375,   250,   390,   291,   453,   251,   632,   459,   461,
     463,   465,   467,   469,   471,   473,   444,   385,   252,   445,
     482,   484,   486,   305,   307,   309,   311,   313,   659,   716,
     504,   507,   717,   253,   514,   377,   254,   333,   335,   376,
     338,   391,   264,   343,   345,   292,   265,   266,   267,   268,
     293,   294,   295,   296,   297,   264,   255,   298,   292,   265,
     266,   267,   268,   293,   294,   295,   296,   297,   381,   399,
     401,   403,   405,   407,   409,   411,   378,   392,   539,   540,
     393,   304,   256,   336,   264,  -336,   434,   724,   265,   266,
     267,   268,   383,   257,   455,   457,   395,   306,   339,   396,
    -338,   424,   258,  -342,   475,   477,   479,   259,   260,   382,
     308,   488,   490,   492,   494,   496,   498,   500,   394,   264,
     510,   512,   292,   265,   266,   267,   268,   293,   294,   295,
     296,   297,   310,   384,   298,   264,   261,   397,   292,   265,
     266,   267,   268,   293,   294,   295,   296,   297,   264,   312,
     298,   292,   265,   266,   267,   268,   293,   294,   295,   296,
     297,   264,   332,   298,   726,   265,   266,   267,   268,   480,
     264,  -346,   414,   292,   265,   266,   267,   268,   293,   294,
     295,   296,   297,   425,   262,   298,   426,   264,   416,   418,
     292,   265,   266,   267,   268,   293,   294,   295,   296,   297,
     264,   263,   298,   292,   265,   266,   267,   268,   293,   294,
     295,   296,   297,   415,   264,   298,   277,   728,   265,   266,
     267,   268,   523,   420,   427,     2,     3,   506,     4,   417,
     419,   278,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,   624,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,   421,   264,    36,   645,   279,   265,
     266,   267,   268,   280,   502,   503,   282,    37,    38,   284,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   334,   661,   663,   665,   667,   669,   671,   673,   675,
     677,   679,   428,   285,   437,   429,   286,   438,   515,   446,
     657,   519,   530,   449,   518,   292,   528,   422,   532,   431,
     293,   294,   295,   296,   297,   534,   684,   521,   536,   264,
     537,   523,   292,   265,   266,   267,   268,   293,   294,   295,
     296,   297,   287,   430,   298,   439,   547,   447,   538,   523,
     448,   450,   718,   784,   451,   721,   722,   723,   423,   387,
     432,   541,   732,   733,   734,   542,   660,   662,   664,   666,
     668,   670,   672,   674,   676,   678,   543,   544,   337,   545,
     264,   548,   622,   549,   265,   266,   267,   268,   550,   551,
     552,   695,   696,   698,   700,   701,   702,   264,   697,   623,
     292,   265,   266,   267,   268,   293,   294,   295,   296,   297,
     553,   554,   725,   727,   729,   731,   264,   289,   555,   292,
     265,   266,   267,   268,   293,   294,   295,   296,   297,   556,
     557,   298,   558,   264,   559,   560,   561,   265,   266,   267,
     268,   646,   647,   648,   649,   650,   651,   652,   653,   654,
     655,   656,   562,   563,   564,   264,   314,   565,   566,   265,
     266,   267,   268,   567,   680,   681,   568,   569,   570,   685,
     686,   687,   688,   689,   690,   691,   692,   693,   694,   342,
     571,   572,   573,   625,   264,   703,   704,   705,   265,   266,
     267,   268,   344,   574,   264,   719,   720,   575,   265,   266,
     267,   268,   576,   577,   578,   316,   579,   735,   736,   737,
     738,   739,   740,   741,   742,   580,   581,   264,   582,   583,
     292,   265,   266,   267,   268,   293,   294,   295,   296,   297,
     264,   398,   298,   292,   265,   266,   267,   268,   293,   294,
     295,   296,   297,   264,   400,   298,   584,   265,   266,   267,
     268,   272,   768,   769,   273,   585,   586,   318,   774,   775,
     274,   275,   778,   779,   587,   402,   276,   588,   589,   264,
     590,   591,   292,   265,   266,   267,   268,   293,   294,   295,
     296,   297,   264,   404,   298,   292,   265,   266,   267,   268,
     293,   294,   295,   296,   297,   264,   320,   298,   592,   265,
     266,   267,   268,   264,   593,   516,   292,   265,   266,   267,
     268,   293,   294,   295,   296,   297,   594,   595,   298,   596,
     597,   264,   598,   599,   292,   265,   266,   267,   268,   293,
     294,   295,   296,   297,   264,   600,   298,   601,   265,   266,
     267,   268,   770,   771,   772,   773,   602,   603,   776,   777,
     604,   605,   780,   781,    37,    38,   606,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   106,   107,   108,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,   406,   607,
     292,   608,   626,   609,   610,   293,   294,   295,   296,   297,
     264,   408,   521,   292,   265,   266,   267,   268,   293,   294,
     295,   296,   297,   641,   410,   298,   292,   611,   627,   612,
     613,   293,   294,   295,   296,   297,   264,   433,   521,   292,
     265,   266,   267,   268,   293,   294,   295,   296,   297,   264,
     454,   298,   292,   265,   266,   267,   268,   293,   294,   295,
     296,   297,   264,   456,   298,   292,   265,   266,   267,   268,
     293,   294,   295,   296,   297,   264,   474,   298,   292,   265,
     266,   267,   268,   293,   294,   295,   296,   297,   264,   476,
     298,   292,   265,   266,   267,   268,   293,   294,   295,   296,
     297,   264,   478,   298,   292,   265,   266,   267,   268,   293,
     294,   295,   296,   297,   264,   487,   298,   292,   265,   266,
     267,   268,   293,   294,   295,   296,   297,   264,   489,   298,
     292,   265,   266,   267,   268,   293,   294,   295,   296,   297,
     264,   491,   298,   292,   265,   266,   267,   268,   293,   294,
     295,   296,   297,   264,   493,   298,   292,   265,   266,   267,
     268,   293,   294,   295,   296,   297,   264,   495,   298,   292,
     265,   266,   267,   268,   293,   294,   295,   296,   297,   264,
     497,   298,   292,   265,   266,   267,   268,   293,   294,   295,
     296,   297,   264,   499,   298,   292,   265,   266,   267,   268,
     293,   294,   295,   296,   297,   264,   509,   298,   292,   265,
     266,   267,   268,   293,   294,   295,   296,   297,   264,   511,
     298,   292,   265,   266,   267,   268,   293,   294,   295,   296,
     297,   264,   322,   298,   292,   265,   266,   267,   268,   293,
     294,   295,   296,   297,   264,   324,   298,   292,   265,   266,
     267,   268,   293,   294,   295,   296,   297,   264,   326,   298,
     292,   265,   266,   267,   268,   293,   294,   295,   296,   297,
     264,   328,   298,   614,   265,   266,   267,   268,   615,   330,
     616,   617,   618,   264,   619,   620,   621,   265,   266,   267,
     268,   346,   628,   633,   743,   634,   264,   635,   636,   348,
     265,   266,   267,   268,   637,   638,   639,   350,   640,   264,
     744,   682,   746,   265,   266,   267,   268,   264,   352,   745,
     747,   265,   266,   267,   268,   683,   354,   748,   749,   264,
     782,   750,   751,   265,   266,   267,   268,   264,   356,   752,
     753,   265,   266,   267,   268,   264,   358,   754,   755,   265,
     266,   267,   268,   756,   360,   757,   264,   758,   783,   759,
     265,   266,   267,   268,   264,   362,   760,   761,   265,   266,
     267,   268,   762,   364,   763,   764,   264,   508,   765,   766,
     265,   266,   267,   268,   264,   366,   785,   788,   265,   266,
     267,   268,   264,   379,   789,   790,   265,   266,   267,   268,
     791,   412,   792,   264,   767,   793,   517,   265,   266,   267,
     268,   264,   435,   787,     0,   265,   266,   267,   268,     0,
     452,     0,     0,   264,     0,     0,     0,   265,   266,   267,
     268,   264,   458,     0,     0,   265,   266,   267,   268,   264,
     460,     0,     0,   265,   266,   267,   268,     0,   462,     0,
     264,     0,     0,     0,   265,   266,   267,   268,   264,   464,
       0,     0,   265,   266,   267,   268,     0,   466,     0,     0,
     264,     0,     0,     0,   265,   266,   267,   268,   264,   468,
       0,     0,   265,   266,   267,   268,   264,   470,     0,     0,
     265,   266,   267,   268,     0,   472,     0,   264,     0,     0,
       0,   265,   266,   267,   268,   264,   481,     0,     0,   265,
     266,   267,   268,     0,   483,     0,     0,   264,     0,     0,
       0,   265,   266,   267,   268,   264,   485,     0,     0,   265,
     266,   267,   268,   264,   513,     0,     0,   265,   266,   267,
     268,     0,     0,     0,   264,   699,     0,     0,   265,   266,
     267,   268,   264,     0,     0,     0,   265,   266,   267,   268,
       0,     0,     0,     0,   264,     0,     0,     0,   265,   266,
     267,   268,   264,     0,     0,     0,   265,   266,   267,   268,
     264,     0,     0,     0,   265,   266,   267,   268,   264,     0,
       0,   292,   265,   266,   267,   268,   293,   294,   295,   296,
     297,   264,     0,     0,   292,   265,   266,   267,   268,   264,
     294,   295,   730,   265,   266,   267,   268
};

static const yytype_int16 yycheck[] =
{
      25,    26,    81,    82,   248,   249,     3,    39,   248,   249,
      42,   255,    37,    38,     1,   255,     4,     4,     1,     1,
      45,    46,    47,    48,    49,    50,    51,    52,    53,     1,
      39,     4,    39,    42,     1,    42,     4,     4,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,     4,
     250,   251,   252,    78,   254,    42,    39,   257,     1,    42,
      42,   261,    39,    39,     4,    42,    42,    39,    93,    42,
       4,    43,    44,    45,    46,    42,    48,    49,     1,   104,
       1,     4,     4,     4,     1,   110,     4,    42,   113,   114,
     115,   116,   117,   118,   119,   120,    39,     1,     4,    42,
     125,   126,   127,    40,    41,    42,    43,    44,    42,    39,
     135,   136,    42,     4,   139,     1,     4,    54,    55,    42,
      57,    42,    39,    60,    61,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     4,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,     1,    86,
      87,    88,    89,    90,    91,    92,    42,     1,     5,     6,
       4,     1,     4,     1,    39,     3,   103,    42,    43,    44,
      45,    46,     1,     4,   111,   112,     1,     1,     1,     4,
       3,     1,     4,     3,   121,   122,   123,    39,     4,    42,
       1,   128,   129,   130,   131,   132,   133,   134,    42,    39,
     137,   138,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,     1,    42,    54,    39,     4,    42,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,     1,
      54,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,     1,
      39,     3,     1,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,     1,     4,    54,     4,    39,     1,     1,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     4,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    42,    39,    54,    42,    42,    43,    44,
      45,    46,   546,     1,    42,     0,     1,     1,     3,    42,
      42,    48,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,   522,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    42,    39,    41,   547,    48,    43,
      44,    45,    46,    39,    48,    49,    39,    52,    53,    39,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,     1,   561,   562,   563,   564,   565,   566,   567,   568,
     569,   570,     1,    39,     1,     4,     6,     4,     3,     1,
     559,    42,    39,     1,     3,    42,    40,     1,    39,     1,
      47,    48,    49,    50,    51,    40,   575,    54,     5,    39,
       5,   745,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,     1,    42,    54,    42,     4,    39,     5,   763,
      42,    39,   601,   763,    42,   604,   605,   606,    42,     1,
      42,     5,   611,   612,   613,     5,   561,   562,   563,   564,
     565,   566,   567,   568,   569,   570,     5,     5,     1,     5,
      39,     5,     3,     5,    43,    44,    45,    46,     5,     5,
       5,   586,   587,   588,   589,   590,   591,    39,     4,    40,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
       5,     5,   607,   608,   609,   610,    39,     1,     5,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,     5,
       5,    54,     5,    39,     5,     5,     5,    43,    44,    45,
      46,   548,   549,   550,   551,   552,   553,   554,   555,   556,
     557,   558,     5,     5,     5,    39,     1,     5,     5,    43,
      44,    45,    46,     5,   571,   572,     5,     5,     5,   576,
     577,   578,   579,   580,   581,   582,   583,   584,   585,     1,
       5,     5,     5,    40,    39,   592,   593,   594,    43,    44,
      45,    46,     1,     5,    39,   602,   603,     5,    43,    44,
      45,    46,     5,     5,     5,     1,     5,   614,   615,   616,
     617,   618,   619,   620,   621,     5,     5,    39,     5,     5,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,    54,     5,    43,    44,    45,
      46,    39,   747,   748,    42,     5,     5,     1,   753,   754,
      48,    49,   757,   758,     5,     1,    54,     5,     5,    39,
       5,     5,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     1,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     1,    54,     5,    43,
      44,    45,    46,    39,     5,     3,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,     5,     5,    54,     5,
       5,    39,     5,     5,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    39,     5,    54,     5,    43,    44,
      45,    46,   749,   750,   751,   752,     5,     5,   755,   756,
       5,     5,   759,   760,    52,    53,     5,    55,    56,    57,
      58,    59,    60,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,   106,   107,
     108,   109,   110,   111,   112,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,   140,   141,   142,   143,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   153,   154,   155,     1,     5,
      42,     5,    40,     5,     5,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,    54,    42,     5,    40,     5,
       5,    47,    48,    49,    50,    51,    39,     1,    54,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    39,
       1,    54,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     1,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     1,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,     1,
      54,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    39,     1,    54,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    39,     1,    54,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    39,     1,    54,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    39,     1,    54,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    39,
       1,    54,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    39,     1,    54,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    39,     1,    54,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    39,     1,
      54,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,     1,    54,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    39,     1,    54,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    39,     1,    54,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      39,     1,    54,     5,    43,    44,    45,    46,     5,     1,
       5,     5,     5,    39,     5,     5,     5,    43,    44,    45,
      46,     1,    40,     4,    40,     4,    39,    42,     4,     1,
      43,    44,    45,    46,     4,     4,    39,     1,    39,    39,
       5,     4,    40,    43,    44,    45,    46,    39,     1,     5,
       5,    43,    44,    45,    46,    42,     1,     5,     5,    39,
      42,     5,     5,    43,    44,    45,    46,    39,     1,     5,
       5,    43,    44,    45,    46,    39,     1,     5,     5,    43,
      44,    45,    46,     5,     1,     5,    39,     5,    42,     5,
      43,    44,    45,    46,    39,     1,     5,     5,    43,    44,
      45,    46,     5,     1,     5,     5,    39,   136,     5,     4,
      43,    44,    45,    46,    39,     1,    42,     5,    43,    44,
      45,    46,    39,     1,     5,     4,    43,    44,    45,    46,
      42,     1,     5,    39,   745,     4,   142,    43,    44,    45,
      46,    39,     1,   765,    -1,    43,    44,    45,    46,    -1,
       1,    -1,    -1,    39,    -1,    -1,    -1,    43,    44,    45,
      46,    39,     1,    -1,    -1,    43,    44,    45,    46,    39,
       1,    -1,    -1,    43,    44,    45,    46,    -1,     1,    -1,
      39,    -1,    -1,    -1,    43,    44,    45,    46,    39,     1,
      -1,    -1,    43,    44,    45,    46,    -1,     1,    -1,    -1,
      39,    -1,    -1,    -1,    43,    44,    45,    46,    39,     1,
      -1,    -1,    43,    44,    45,    46,    39,     1,    -1,    -1,
      43,    44,    45,    46,    -1,     1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,     1,    -1,    -1,    43,
      44,    45,    46,    -1,     1,    -1,    -1,    39,    -1,    -1,
      -1,    43,    44,    45,    46,    39,     1,    -1,    -1,    43,
      44,    45,    46,    39,     1,    -1,    -1,    43,    44,    45,
      46,    -1,    -1,    -1,    39,     4,    -1,    -1,    43,    44,
      45,    46,    39,    -1,    -1,    -1,    43,    44,    45,    46,
      -1,    -1,    -1,    -1,    39,    -1,    -1,    -1,    43,    44,
      45,    46,    39,    -1,    -1,    -1,    43,    44,    45,    46,
      39,    -1,    -1,    -1,    43,    44,    45,    46,    39,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    39,    -1,    -1,    42,    43,    44,    45,    46,    39,
      48,    49,    42,    43,    44,    45,    46
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   157,     0,     1,     3,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    41,    52,    53,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
     106,   107,   108,   109,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   154,   155,
     158,   167,   171,   175,   176,   177,   178,   179,   180,   181,
     182,   183,   184,   185,   186,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   214,   215,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   229,   230,   231,
     232,   233,   234,   235,   236,   237,   238,   239,   240,   241,
     242,   243,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     3,     4,     4,     4,
       4,     4,     4,     4,     4,     4,     4,     4,     4,    39,
       4,     4,     4,     4,    39,    43,    44,    45,    46,   161,
     162,   161,    39,    42,    48,    49,    54,    42,    48,    48,
      39,   169,    39,   170,    39,    39,     6,     1,   161,     1,
     161,     1,    42,    47,    48,    49,    50,    51,    54,   159,
     160,   161,   164,   165,     1,   159,     1,   159,     1,   159,
       1,   159,     1,   159,     1,   161,     1,   161,     1,   161,
       1,   161,     1,   161,     1,   161,     1,   161,     1,   161,
       1,   161,     1,   159,     1,   159,     1,     1,   159,     1,
       1,    42,     1,   159,     1,   159,     1,   161,     1,   161,
       1,   161,     1,   161,     1,   161,     1,   161,     1,   161,
       1,   161,     1,   161,     1,   161,     1,   161,     1,     4,
      42,     1,     4,    42,     1,     4,    42,     1,    42,     1,
     161,     1,    42,     1,    42,     1,   160,     1,   160,     1,
       4,    42,     1,     4,    42,     1,     4,    42,     1,   159,
       1,   159,     1,   159,     1,   159,     1,   159,     1,   159,
       1,   159,     1,   161,     1,    42,     1,    42,     1,    42,
       1,    42,     1,    42,     1,     1,     4,    42,     1,     4,
      42,     1,    42,     1,   159,     1,   161,     1,     4,    42,
       1,    39,    42,     1,    39,    42,     1,    39,    42,     1,
      39,    42,     1,   161,     1,   159,     1,   159,     1,   161,
       1,   161,     1,   161,     1,   161,     1,   161,     1,   161,
       1,   161,     1,   161,     1,   159,     1,   159,     1,   159,
       1,     1,   161,     1,   161,     1,   161,     1,   159,     1,
     159,     1,   159,     1,   159,     1,   159,     1,   159,     1,
     159,     1,    48,    49,   161,   166,     1,   161,   166,     1,
     159,     1,   159,     1,   161,     3,     3,   175,     3,    42,
     174,    54,   163,   164,   163,   174,   174,   174,    40,   174,
      39,   163,    39,   174,    40,   174,     5,     5,     5,     5,
       6,     5,     5,     5,     5,     5,   168,     4,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     3,    40,   174,    40,    40,    40,    40,     4,
      42,     4,    42,     4,     4,    42,     4,     4,     4,    39,
      39,    39,   163,   172,   173,   174,   159,   159,   159,   159,
     159,   159,   159,   159,   159,   159,   159,   160,     4,    42,
     161,   165,   161,   165,   161,   165,   161,   165,   161,   165,
     161,   165,   161,   165,   161,   165,   161,   165,   161,   165,
     159,   159,     4,    42,   160,   159,   159,   159,   159,   159,
     159,   159,   159,   159,   159,   161,   161,     4,   161,     4,
     161,   161,   161,   159,   159,   159,    39,    42,    39,    42,
      39,    42,    39,    42,    39,    42,    39,    42,   160,   159,
     159,   160,   160,   160,    42,   161,    42,   161,    42,   161,
      42,   161,   160,   160,   160,   159,   159,   159,   159,   159,
     159,   159,   159,    40,     5,     5,    40,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     5,     5,     5,     5,
       5,     5,     5,     5,     5,     5,     4,   173,   161,   161,
     159,   159,   159,   159,   161,   161,   159,   159,   161,   161,
     159,   159,    42,    42,   163,    42,   244,   244,     5,     5,
       4,    42,     5,     4
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 9:
#line 237 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax, LINE - 1 ); ;}
    break;

  case 31:
#line 251 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addEntry(); ;}
    break;

  case 32:
#line 252 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->setModuleName( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 33:
#line 253 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 34:
#line 254 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addGlobal( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 35:
#line 255 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 36:
#line 256 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addVar( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 37:
#line 257 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 38:
#line 258 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addConst( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 39:
#line 259 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 40:
#line 260 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addAttrib( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 41:
#line 261 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLocal( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 42:
#line 262 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addParam( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 43:
#line 263 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 44:
#line 264 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFuncDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 45:
#line 265 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 46:
#line 266 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFunction( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 47:
#line 267 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); ;}
    break;

  case 48:
#line 268 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addLoad( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 49:
#line 269 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 50:
#line 270 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 51:
#line 271 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 52:
#line 272 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDSwitch( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]), true ); ;}
    break;

  case 53:
#line 273 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 54:
#line 274 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 55:
#line 275 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 56:
#line 276 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 57:
#line 277 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 58:
#line 278 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDCase( (yyvsp[(2) - (6)]), (yyvsp[(6) - (6)]), (yyvsp[(4) - (6)]) ); ;}
    break;

  case 59:
#line 279 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addDEndSwitch(); ;}
    break;

  case 60:
#line 280 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 61:
#line 281 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addProperty( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 62:
#line 282 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addPropRef( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 65:
#line 285 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 66:
#line 286 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstance( (yyvsp[(2) - (5)]), (yyvsp[(3) - (5)]), (yyvsp[(4) - (5)]), true ); ;}
    break;

  case 67:
#line 287 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 68:
#line 288 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClass( (yyvsp[(2) - (4)]), (yyvsp[(3) - (4)]), true ); ;}
    break;

  case 69:
#line 289 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 70:
#line 290 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassDef( (yyvsp[(2) - (3)]), true ); ;}
    break;

  case 71:
#line 291 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addClassCtor( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 72:
#line 292 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addFuncEnd(); /* Currently the same as .endfunc */ ;}
    break;

  case 73:
#line 293 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInherit((yyvsp[(2) - (2)])); ;}
    break;

  case 75:
#line 294 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addFrom( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 76:
#line 295 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addExtern( (yyvsp[(2) - (3)]), (yyvsp[(3) - (3)]) ); ;}
    break;

  case 77:
#line 296 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addDLine( (yyvsp[(2) - (2)]) ); ;}
    break;

  case 78:
#line 298 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 79:
#line 303 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         // string already added to the module by the lexer
         delete (yyvsp[(2) - (2)]);
      ;}
    break;

  case 80:
#line 311 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 81:
#line 312 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHas( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 82:
#line 316 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 83:
#line 317 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->classHasnt( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 84:
#line 320 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->defineLabel( (yyvsp[(1) - (2)])->asLabel() ); ;}
    break;

  case 86:
#line 324 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(1) - (1)]) ); ;}
    break;

  case 87:
#line 325 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInheritParam( (yyvsp[(3) - (3)]) ); ;}
    break;

  case 90:
#line 334 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {(yyval) = new Falcon::Pseudo( LINE, (Falcon::int64) 0 ); ;}
    break;

  case 194:
#line 444 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 195:
#line 445 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LD" ); ;}
    break;

  case 196:
#line 449 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDRF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 197:
#line 450 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDRF" ); ;}
    break;

  case 198:
#line 454 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LNIL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 199:
#line 455 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LNIL" ); ;}
    break;

  case 200:
#line 459 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 201:
#line 460 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADD" ); ;}
    break;

  case 202:
#line 464 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ADDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 203:
#line 465 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ADDS" ); ;}
    break;

  case 204:
#line 470 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 205:
#line 471 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUB" ); ;}
    break;

  case 206:
#line 475 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SUBS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 207:
#line 476 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SUBS" ); ;}
    break;

  case 208:
#line 480 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MUL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 209:
#line 481 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MUL" ); ;}
    break;

  case 210:
#line 485 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MULS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 211:
#line 486 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MULS" ); ;}
    break;

  case 212:
#line 491 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 213:
#line 492 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIV" ); ;}
    break;

  case 214:
#line 496 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DIVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 215:
#line 497 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DIVS" ); ;}
    break;

  case 216:
#line 501 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MOD, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 217:
#line 502 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MOD" ); ;}
    break;

  case 218:
#line 506 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POW, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 219:
#line 507 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POW" ); ;}
    break;

  case 220:
#line 512 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_EQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 221:
#line 513 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "EQ" ); ;}
    break;

  case 222:
#line 517 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEQ, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 223:
#line 518 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEQ" ); ;}
    break;

  case 224:
#line 522 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 225:
#line 523 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GE" ); ;}
    break;

  case 226:
#line 527 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 227:
#line 528 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GT" ); ;}
    break;

  case 228:
#line 532 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 229:
#line 533 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LE" ); ;}
    break;

  case 230:
#line 537 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 231:
#line 538 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LT" ); ;}
    break;

  case 232:
#line 542 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 233:
#line 543 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed(true); COMPILER->addInstr( P_TRY, (yyvsp[(2) - (2)])); ;}
    break;

  case 234:
#line 544 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRY" ); ;}
    break;

  case 235:
#line 548 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 236:
#line 549 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INC" ); ;}
    break;

  case 237:
#line 553 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DEC, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 238:
#line 554 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DEC" ); ;}
    break;

  case 239:
#line 559 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INCP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 240:
#line 560 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INCP" ); ;}
    break;

  case 241:
#line 564 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_DECP, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 242:
#line 565 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "DECP" ); ;}
    break;

  case 243:
#line 570 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NEG, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 244:
#line 571 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NEG" ); ;}
    break;

  case 245:
#line 575 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOT, (yyvsp[(2) - (2)])  ); ;}
    break;

  case 246:
#line 576 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOT" ); ;}
    break;

  case 247:
#line 580 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 248:
#line 581 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_CALL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 249:
#line 582 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "CALL" ); ;}
    break;

  case 250:
#line 586 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 251:
#line 587 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_INST, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 252:
#line 588 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INST" ); ;}
    break;

  case 253:
#line 592 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_UNPK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 254:
#line 593 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPK" ); ;}
    break;

  case 255:
#line 597 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_UNPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 256:
#line 598 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "UNPS" ); ;}
    break;

  case 257:
#line 603 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PUSH, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 258:
#line 604 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHN ); ;}
    break;

  case 259:
#line 605 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PUSH" ); ;}
    break;

  case 260:
#line 609 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSHR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 261:
#line 610 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSHR" ); ;}
    break;

  case 262:
#line 615 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_POP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 263:
#line 616 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POP" ); ;}
    break;

  case 264:
#line 620 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {  COMPILER->addInstr( P_PEEK, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 265:
#line 621 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PEEK" ); ;}
    break;

  case 266:
#line 625 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 267:
#line 626 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XPOP" ); ;}
    break;

  case 268:
#line 631 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 269:
#line 632 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 270:
#line 633 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDV" ); ;}
    break;

  case 271:
#line 637 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 272:
#line 638 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 273:
#line 639 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVT" ); ;}
    break;

  case 274:
#line 643 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 275:
#line 644 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 276:
#line 645 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STV" ); ;}
    break;

  case 277:
#line 649 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 278:
#line 650 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 279:
#line 651 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVR" ); ;}
    break;

  case 280:
#line 655 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 281:
#line 656 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STVS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 282:
#line 657 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STVS" ); ;}
    break;

  case 283:
#line 661 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 284:
#line 662 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDP, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 285:
#line 663 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDP" ); yyerrok; ;}
    break;

  case 286:
#line 667 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 287:
#line 668 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPT, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 288:
#line 669 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPT" ); yyerrok; ;}
    break;

  case 289:
#line 673 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 290:
#line 674 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STP, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 291:
#line 675 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STP" ); ;}
    break;

  case 292:
#line 679 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 293:
#line 680 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 294:
#line 681 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPR" ); ;}
    break;

  case 295:
#line 685 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 296:
#line 686 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STPS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 297:
#line 687 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STPS" ); ;}
    break;

  case 298:
#line 691 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 299:
#line 692 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAV, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 300:
#line 693 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAV" ); ;}
    break;

  case 301:
#line 697 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 302:
#line 698 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed( true ); (yyvsp[(4) - (6)])->fixed( true ); (yyvsp[(6) - (6)])->fixed( true ); COMPILER->addInstr( P_TRAN, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 303:
#line 699 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAN" ); ;}
    break;

  case 304:
#line 703 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 305:
#line 704 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_TRAL, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 306:
#line 705 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "TRAL" ); ;}
    break;

  case 307:
#line 709 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_IPOP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 308:
#line 710 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IPOP" ); ;}
    break;

  case 309:
#line 714 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GENA, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 310:
#line 715 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENA" ); ;}
    break;

  case 311:
#line 719 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_GEND, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 312:
#line 720 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEND" ); ;}
    break;

  case 313:
#line 724 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GENR, (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)])); ;}
    break;

  case 314:
#line 725 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GENR" ); ;}
    break;

  case 315:
#line 729 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GEOR, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 316:
#line 730 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GEOR" ); ;}
    break;

  case 317:
#line 734 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RIS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 318:
#line 735 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RIS" ); ;}
    break;

  case 319:
#line 739 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 320:
#line 740 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JMP, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 321:
#line 741 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JMP" ); ;}
    break;

  case 322:
#line 745 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOOL, (yyvsp[(1) - (2)]) ); ;}
    break;

  case 323:
#line 746 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOOL" ); ;}
    break;

  case 324:
#line 750 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 325:
#line 751 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFT, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 326:
#line 752 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFT" ); ;}
    break;

  case 327:
#line 756 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 328:
#line 757 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_IFF, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 329:
#line 758 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IFF" ); ;}
    break;

  case 330:
#line 763 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 331:
#line 764 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); (yyvsp[(4) - (4)])->fixed( true ); COMPILER->addInstr( P_FORK, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 332:
#line 765 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "FORK" ); ;}
    break;

  case 333:
#line 769 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 334:
#line 770 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_JTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 335:
#line 771 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "JTRY" ); ;}
    break;

  case 336:
#line 775 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RET ); ;}
    break;

  case 337:
#line 776 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RET" ); ;}
    break;

  case 338:
#line 780 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETA ); ;}
    break;

  case 339:
#line 781 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETA" ); ;}
    break;

  case 340:
#line 785 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_RETV, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 341:
#line 786 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "RETV" ); ;}
    break;

  case 342:
#line 790 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOP ); ;}
    break;

  case 343:
#line 791 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOP" ); ;}
    break;

  case 344:
#line 795 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (2)])->fixed( true ); COMPILER->addInstr( P_PTRY, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 345:
#line 796 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PTRY" ); ;}
    break;

  case 346:
#line 800 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_END ); ;}
    break;

  case 347:
#line 801 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "END" ); ;}
    break;

  case 348:
#line 805 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 349:
#line 806 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SWCH" ); ;}
    break;

  case 350:
#line 810 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (6)])->fixed(true); COMPILER->write_switch( (yyvsp[(2) - (6)]), (yyvsp[(4) - (6)]), (yyvsp[(6) - (6)]) ); ;}
    break;

  case 351:
#line 811 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SELE" ); ;}
    break;

  case 352:
#line 816 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         Falcon::Pseudo *psd = new Falcon::Pseudo( Falcon::Pseudo::tswitch_list );
         psd->line( LINE );
         psd->asList()->pushBack( (yyvsp[(1) - (3)]) );
         psd->asList()->pushBack( (yyvsp[(3) - (3)]) );
         (yyval) = psd;
      ;}
    break;

  case 353:
#line 825 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    {
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(3) - (5)]) );
         (yyvsp[(1) - (5)])->asList()->pushBack( (yyvsp[(5) - (5)]) );
         (yyval) = (yyvsp[(1) - (5)]);
      ;}
    break;

  case 354:
#line 833 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 355:
#line 834 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { (yyvsp[(2) - (4)])->fixed( true ); COMPILER->addInstr( P_ONCE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); COMPILER->addStatic(); ;}
    break;

  case 356:
#line 835 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ONCE" ); ;}
    break;

  case 357:
#line 839 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 358:
#line 840 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 359:
#line 841 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 360:
#line 842 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BAND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 361:
#line 843 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BAND" ); ;}
    break;

  case 362:
#line 847 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 363:
#line 848 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 364:
#line 849 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 365:
#line 850 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 366:
#line 851 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BOR" ); ;}
    break;

  case 367:
#line 855 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 368:
#line 856 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 369:
#line 857 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 370:
#line 858 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BXOR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 371:
#line 859 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 372:
#line 863 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 373:
#line 864 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_BNOT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 374:
#line 865 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "BXOR" ); ;}
    break;

  case 375:
#line 869 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_AND, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 376:
#line 870 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "AND" ); ;}
    break;

  case 377:
#line 874 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_OR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 378:
#line 875 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "OR" ); ;}
    break;

  case 379:
#line 879 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ANDS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 380:
#line 880 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ANDS" ); ;}
    break;

  case 381:
#line 884 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_ORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 382:
#line 885 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "ORS" ); ;}
    break;

  case 383:
#line 889 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_XORS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 384:
#line 890 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "XORS" ); ;}
    break;

  case 385:
#line 894 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_MODS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 386:
#line 895 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "MODS" ); ;}
    break;

  case 387:
#line 899 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_POWS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 388:
#line 900 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "POWS" ); ;}
    break;

  case 389:
#line 904 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOTS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 390:
#line 905 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOTS" ); ;}
    break;

  case 391:
#line 909 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 392:
#line 910 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HAS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 393:
#line 911 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HAS" ); ;}
    break;

  case 394:
#line 915 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 395:
#line 916 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_HASN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 396:
#line 917 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "HASN" ); ;}
    break;

  case 397:
#line 921 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 398:
#line 922 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVE, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 399:
#line 923 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVE" ); ;}
    break;

  case 400:
#line 927 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 401:
#line 928 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_GIVN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 402:
#line 929 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "GIVN" ); ;}
    break;

  case 403:
#line 934 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_IN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 404:
#line 935 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "IN" ); ;}
    break;

  case 405:
#line 939 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_NOIN, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 406:
#line 940 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "NOIN" ); ;}
    break;

  case 407:
#line 944 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PROV, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)])); ;}
    break;

  case 408:
#line 945 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PROV" ); ;}
    break;

  case 409:
#line 949 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PSIN, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 410:
#line 950 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PSIN" ); ;}
    break;

  case 411:
#line 954 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_PASS, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 412:
#line 955 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "PASS" ); ;}
    break;

  case 413:
#line 959 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 414:
#line 960 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHR" ); ;}
    break;

  case 415:
#line 964 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHL, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 416:
#line 965 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHL" ); ;}
    break;

  case 417:
#line 969 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHRS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 418:
#line 970 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHRS" ); ;}
    break;

  case 419:
#line 974 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_SHLS, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 420:
#line 975 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "SHLS" ); ;}
    break;

  case 421:
#line 979 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDVR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 422:
#line 980 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDVR" ); ;}
    break;

  case 423:
#line 984 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LDPR, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 424:
#line 985 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LDPR" ); ;}
    break;

  case 425:
#line 989 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_LSB, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 426:
#line 990 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "LSB" ); ;}
    break;

  case 427:
#line 994 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 428:
#line 995 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_INDI, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 429:
#line 996 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "INDI" ); ;}
    break;

  case 430:
#line 1000 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 431:
#line 1001 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STEX, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 432:
#line 1002 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "STEX" ); ;}
    break;

  case 433:
#line 1006 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_TRAC, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 434:
#line 1007 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "TRAC" ); ;}
    break;

  case 435:
#line 1011 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_WRT, (yyvsp[(2) - (2)]) ); ;}
    break;

  case 436:
#line 1012 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError( Falcon::e_invop, "WRT" ); ;}
    break;

  case 437:
#line 1017 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->addInstr( P_STO, (yyvsp[(2) - (4)]), (yyvsp[(4) - (4)]) ); ;}
    break;

  case 438:
#line 1018 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
    { COMPILER->raiseError(Falcon::e_invop, "STO" ); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 4120 "/home/gian/Progetti/falcon/core/engine/fasm_parser.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1022 "/home/gian/Progetti/falcon/core/engine/fasm_parser.yy"
 /* c code */


/****************************************************
* C Code for falcon HSM compiler
*****************************************************/


void fasm_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of falcon_parser.yxx */



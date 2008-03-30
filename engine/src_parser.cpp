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
#define yyparse flc_src_parse
#define yylex   flc_src_lex
#define yyerror flc_src_error
#define yylval  flc_src_lval
#define yychar  flc_src_char
#define yydebug flc_src_debug
#define yynerrs flc_src_nerrs


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
     ASSIGN_POW = 319,
     ASSIGN_SHL = 320,
     ASSIGN_SHR = 321,
     ASSIGN_BXOR = 322,
     ASSIGN_BOR = 323,
     ASSIGN_BAND = 324,
     ASSIGN_MOD = 325,
     ASSIGN_DIV = 326,
     ASSIGN_MUL = 327,
     ASSIGN_SUB = 328,
     ASSIGN_ADD = 329,
     OP_EQ = 330,
     OP_TO = 331,
     COMMA = 332,
     QUESTION = 333,
     OR = 334,
     AND = 335,
     NOT = 336,
     LE = 337,
     GE = 338,
     LT = 339,
     GT = 340,
     NEQ = 341,
     EEQ = 342,
     PROVIDES = 343,
     OP_NOTIN = 344,
     OP_IN = 345,
     HASNT = 346,
     HAS = 347,
     DIESIS = 348,
     ATSIGN = 349,
     CAP = 350,
     VBAR = 351,
     AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     BANG = 361,
     NEG = 362,
     DECREMENT = 363,
     INCREMENT = 364,
     DOLLAR = 365
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
#define ASSIGN_POW 319
#define ASSIGN_SHL 320
#define ASSIGN_SHR 321
#define ASSIGN_BXOR 322
#define ASSIGN_BOR 323
#define ASSIGN_BAND 324
#define ASSIGN_MOD 325
#define ASSIGN_DIV 326
#define ASSIGN_MUL 327
#define ASSIGN_SUB 328
#define ASSIGN_ADD 329
#define OP_EQ 330
#define OP_TO 331
#define COMMA 332
#define QUESTION 333
#define OR 334
#define AND 335
#define NOT 336
#define LE 337
#define GE 338
#define LT 339
#define GT 340
#define NEQ 341
#define EEQ 342
#define PROVIDES 343
#define OP_NOTIN 344
#define OP_IN 345
#define HASNT 346
#define HAS 347
#define DIESIS 348
#define ATSIGN 349
#define CAP 350
#define VBAR 351
#define AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define BANG 361
#define NEG 362
#define DECREMENT 363
#define INCREMENT 364
#define DOLLAR 365




/* Copy the first part of user declarations.  */
#line 22 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


#include <math.h>
#include <stdio.h>
#include <iostream>

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/src_lexer.h>
#include <falcon/syntree.h>
#include <falcon/error.h>
#include <stdlib.h>

#include <falcon/fassert.h>

#include <falcon/memory.h>

#define YYMALLOC	Falcon::memAlloc
#define YYFREE Falcon::memFree

#define  COMPILER  ( reinterpret_cast< Falcon::Compiler *>(yyparam) )
#define  CTX_LINE  ( COMPILER->lexer()->ctxOpenLine() )
#define  LINE      ( COMPILER->lexer()->previousLine() )
#define  CURRENT_LINE      ( COMPILER->lexer()->line() )


#define YYPARSE_PARAM yyparam
#define YYLEX_PARAM yyparam

int flc_src_parse( void *param );
void flc_src_error (const char *s);

inline int flc_src_lex (void *lvalp, void *yyparam)
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
/* Line 187 of yacc.c.  */
#line 380 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 393 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   6150

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  111
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  163
/* YYNRULES -- Number of rules.  */
#define YYNRULES  435
/* YYNRULES -- Number of states.  */
#define YYNSTATES  790

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   365

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
     105,   106,   107,   108,   109,   110
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    63,    67,    71,    74,
      77,    82,    89,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     161,   165,   169,   172,   176,   177,   184,   187,   191,   195,
     199,   203,   204,   206,   207,   211,   214,   218,   219,   224,
     228,   232,   233,   236,   239,   243,   246,   250,   254,   255,
     265,   266,   274,   280,   284,   285,   288,   290,   292,   294,
     296,   300,   304,   308,   311,   315,   318,   322,   326,   328,
     329,   336,   340,   344,   345,   352,   356,   360,   361,   368,
     372,   376,   377,   384,   388,   392,   393,   396,   400,   402,
     403,   409,   410,   416,   417,   423,   424,   430,   431,   432,
     436,   437,   439,   442,   445,   448,   450,   454,   456,   458,
     460,   464,   466,   467,   474,   478,   482,   483,   486,   490,
     492,   493,   499,   500,   506,   507,   513,   514,   520,   522,
     526,   527,   529,   531,   537,   542,   546,   550,   551,   558,
     561,   565,   566,   568,   570,   573,   576,   579,   584,   588,
     594,   598,   600,   604,   606,   608,   612,   616,   622,   625,
     631,   632,   640,   644,   650,   651,   658,   661,   662,   664,
     668,   670,   671,   672,   678,   679,   683,   686,   690,   693,
     697,   701,   705,   709,   715,   721,   725,   731,   737,   741,
     744,   748,   752,   754,   758,   762,   766,   768,   772,   776,
     780,   782,   786,   790,   794,   799,   803,   806,   810,   813,
     817,   818,   820,   824,   827,   831,   834,   835,   844,   848,
     851,   852,   856,   857,   863,   864,   867,   869,   873,   876,
     877,   881,   883,   887,   889,   891,   893,   894,   897,   899,
     901,   903,   905,   906,   914,   920,   925,   926,   930,   934,
     936,   939,   943,   948,   949,   957,   958,   961,   963,   968,
     971,   973,   975,   976,   985,   988,   991,   992,   995,   997,
     999,  1001,  1003,  1004,  1009,  1011,  1015,  1019,  1021,  1024,
    1028,  1032,  1034,  1036,  1038,  1040,  1042,  1044,  1046,  1048,
    1050,  1052,  1054,  1056,  1059,  1063,  1067,  1071,  1075,  1079,
    1083,  1087,  1091,  1095,  1099,  1103,  1106,  1110,  1113,  1116,
    1119,  1122,  1126,  1130,  1134,  1138,  1142,  1146,  1150,  1153,
    1157,  1161,  1165,  1169,  1173,  1176,  1179,  1182,  1185,  1187,
    1189,  1191,  1193,  1195,  1197,  1200,  1202,  1207,  1213,  1217,
    1219,  1221,  1225,  1231,  1235,  1239,  1243,  1247,  1251,  1255,
    1259,  1263,  1267,  1271,  1275,  1279,  1283,  1288,  1293,  1299,
    1307,  1312,  1316,  1317,  1324,  1325,  1332,  1337,  1341,  1344,
    1345,  1352,  1353,  1359,  1361,  1364,  1370,  1376,  1381,  1385,
    1388,  1392,  1396,  1399,  1403,  1407,  1411,  1415,  1420,  1422,
    1426,  1428,  1431,  1433,  1437,  1441
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     112,     0,    -1,   113,    -1,    -1,   113,   114,    -1,   115,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   117,    -1,
     210,    -1,   190,    -1,   218,    -1,   241,    -1,   118,    -1,
     205,    -1,   206,    -1,   208,    -1,   213,    -1,     4,    -1,
      98,     4,    -1,    39,     6,     3,    -1,    39,     7,     3,
      -1,    39,     1,     3,    -1,   119,    -1,     3,    -1,    48,
       1,     3,    -1,    34,     1,     3,    -1,    32,     1,     3,
      -1,     1,     3,    -1,   254,     3,    -1,   270,    75,   254,
       3,    -1,   270,    75,   254,    77,   270,     3,    -1,   121,
      -1,   122,    -1,   139,    -1,   153,    -1,   168,    -1,   126,
      -1,   137,    -1,   138,    -1,   179,    -1,   180,    -1,   189,
      -1,   250,    -1,   246,    -1,   203,    -1,   204,    -1,   144,
      -1,   145,    -1,   146,    -1,   236,    -1,   252,    75,   254,
      -1,   120,    77,   252,    75,   254,    -1,    10,   120,     3,
      -1,    10,     1,     3,    -1,    -1,   124,   123,   136,     9,
       3,    -1,   125,   118,    -1,    11,   254,     3,    -1,    53,
      -1,    11,     1,     3,    -1,    11,   254,    47,    -1,    53,
      47,    -1,    11,     1,    47,    -1,    -1,   128,   127,   136,
     130,     9,     3,    -1,   129,   118,    -1,    15,   254,     3,
      -1,    15,     1,     3,    -1,    15,   254,    47,    -1,    15,
       1,    47,    -1,    -1,   133,    -1,    -1,   132,   131,   136,
      -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,   135,
     134,   136,   130,    -1,    17,   254,     3,    -1,    17,     1,
       3,    -1,    -1,   136,   118,    -1,    12,     3,    -1,    12,
       1,     3,    -1,    13,     3,    -1,    13,    14,     3,    -1,
      13,     1,     3,    -1,    -1,    18,   272,    90,   254,     3,
     140,   142,     9,     3,    -1,    -1,    18,   272,    90,   254,
      47,   141,   118,    -1,    18,   272,    90,     1,     3,    -1,
      18,     1,     3,    -1,    -1,   143,   142,    -1,   118,    -1,
     147,    -1,   149,    -1,   151,    -1,    51,   254,     3,    -1,
      51,     1,     3,    -1,   104,   270,     3,    -1,   104,     3,
      -1,    85,   270,     3,    -1,    85,     3,    -1,   104,     1,
       3,    -1,    85,     1,     3,    -1,    57,    -1,    -1,    19,
       3,   148,   136,     9,     3,    -1,    19,    47,   118,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   150,   136,     9,
       3,    -1,    20,    47,   118,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   152,   136,     9,     3,    -1,    21,    47,
     118,    -1,    21,     1,     3,    -1,    -1,   155,   154,   156,
     162,     9,     3,    -1,    22,   254,     3,    -1,    22,     1,
       3,    -1,    -1,   156,   157,    -1,   156,     1,     3,    -1,
       3,    -1,    -1,    23,   166,     3,   158,   136,    -1,    -1,
      23,   166,    47,   159,   118,    -1,    -1,    23,     1,     3,
     160,   136,    -1,    -1,    23,     1,    47,   161,   118,    -1,
      -1,    -1,   164,   163,   165,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   136,    -1,    47,   118,    -1,   167,    -1,
     166,    77,   167,    -1,     8,    -1,   116,    -1,     7,    -1,
     116,    76,   116,    -1,     6,    -1,    -1,   170,   169,   171,
     162,     9,     3,    -1,    25,   254,     3,    -1,    25,     1,
       3,    -1,    -1,   171,   172,    -1,   171,     1,     3,    -1,
       3,    -1,    -1,    23,   177,     3,   173,   136,    -1,    -1,
      23,   177,    47,   174,   118,    -1,    -1,    23,     1,     3,
     175,   136,    -1,    -1,    23,     1,    47,   176,   118,    -1,
     178,    -1,   177,    77,   178,    -1,    -1,     4,    -1,     6,
      -1,    28,   270,    76,   270,     3,    -1,    28,   270,     1,
       3,    -1,    28,     1,     3,    -1,    29,    47,   118,    -1,
      -1,   182,   181,   136,   183,     9,     3,    -1,    29,     3,
      -1,    29,     1,     3,    -1,    -1,   184,    -1,   185,    -1,
     184,   185,    -1,   186,   136,    -1,    30,     3,    -1,    30,
      90,   252,     3,    -1,    30,   187,     3,    -1,    30,   187,
      90,   252,     3,    -1,    30,     1,     3,    -1,   188,    -1,
     187,    77,   188,    -1,     4,    -1,     6,    -1,    31,   254,
       3,    -1,    31,     1,     3,    -1,   191,   198,   136,     9,
       3,    -1,   193,   118,    -1,   195,    59,   196,    58,     3,
      -1,    -1,   195,    59,   196,     1,   192,    58,     3,    -1,
     195,     1,     3,    -1,   195,    59,   196,    58,    47,    -1,
      -1,   195,    59,     1,   194,    58,    47,    -1,    48,     6,
      -1,    -1,   197,    -1,   196,    77,   197,    -1,     6,    -1,
      -1,    -1,   201,   199,   136,     9,     3,    -1,    -1,   202,
     200,   118,    -1,    49,     3,    -1,    49,     1,     3,    -1,
      49,    47,    -1,    49,     1,    47,    -1,    40,   256,     3,
      -1,    40,     1,     3,    -1,    43,   254,     3,    -1,    43,
     254,    90,   254,     3,    -1,    43,   254,    90,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    75,   251,     3,
      -1,    41,     6,    75,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   207,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   207,    77,     6,    -1,    45,   209,
       3,    -1,    45,     1,     3,    -1,     6,    -1,   207,    77,
       6,    -1,    46,   211,     3,    -1,    46,     1,     3,    -1,
     212,    -1,   211,    77,   212,    -1,     6,    75,     6,    -1,
       6,    75,   116,    -1,   214,   217,     9,     3,    -1,   215,
     216,     3,    -1,    42,     3,    -1,    42,     1,     3,    -1,
      42,    47,    -1,    42,     1,    47,    -1,    -1,     6,    -1,
     216,    77,     6,    -1,   216,     3,    -1,   217,   216,     3,
      -1,     1,     3,    -1,    -1,    32,     6,   219,   220,   229,
     234,     9,     3,    -1,   221,   223,     3,    -1,     1,     3,
      -1,    -1,    59,   196,    58,    -1,    -1,    59,   196,     1,
     222,    58,    -1,    -1,    33,   224,    -1,   225,    -1,   224,
      77,   225,    -1,     6,   226,    -1,    -1,    59,   227,    58,
      -1,   228,    -1,   227,    77,   228,    -1,   251,    -1,     6,
      -1,    27,    -1,    -1,   229,   230,    -1,     3,    -1,   190,
      -1,   233,    -1,   231,    -1,    -1,    38,     3,   232,   198,
     136,     9,     3,    -1,    49,     6,    75,   254,     3,    -1,
       6,    75,   254,     3,    -1,    -1,    92,   235,     3,    -1,
      92,     1,     3,    -1,     6,    -1,    81,     6,    -1,   235,
      77,     6,    -1,   235,    77,    81,     6,    -1,    -1,    54,
       6,   237,     3,   238,     9,     3,    -1,    -1,   238,   239,
      -1,     3,    -1,     6,    75,   251,   240,    -1,     6,   240,
      -1,     3,    -1,    77,    -1,    -1,    34,     6,   242,   243,
     244,   234,     9,     3,    -1,   223,     3,    -1,     1,     3,
      -1,    -1,   244,   245,    -1,     3,    -1,   190,    -1,   233,
      -1,   231,    -1,    -1,    36,   247,   248,     3,    -1,   249,
      -1,   248,    77,   249,    -1,   248,    77,     1,    -1,     6,
      -1,    35,     3,    -1,    35,   254,     3,    -1,    35,     1,
       3,    -1,     8,    -1,    55,    -1,    56,    -1,     4,    -1,
       5,    -1,     7,    -1,     6,    -1,   252,    -1,    27,    -1,
      26,    -1,   251,    -1,   253,    -1,    98,   254,    -1,   254,
      99,   254,    -1,   254,    98,   254,    -1,   254,   102,   254,
      -1,   254,   101,   254,    -1,   254,   100,   254,    -1,   254,
     103,   254,    -1,   254,    97,   254,    -1,   254,    96,   254,
      -1,   254,    95,   254,    -1,   254,   105,   254,    -1,   254,
     104,   254,    -1,   106,   254,    -1,   254,    86,   254,    -1,
     254,   109,    -1,   109,   254,    -1,   254,   108,    -1,   108,
     254,    -1,   254,    87,   254,    -1,   254,    85,   254,    -1,
     254,    84,   254,    -1,   254,    83,   254,    -1,   254,    82,
     254,    -1,   254,    80,   254,    -1,   254,    79,   254,    -1,
      81,   254,    -1,   254,    92,   254,    -1,   254,    91,   254,
      -1,   254,    90,   254,    -1,   254,    89,   254,    -1,   254,
      88,     6,    -1,   110,   252,    -1,   110,   110,    -1,    94,
     254,    -1,    93,   254,    -1,   263,    -1,   258,    -1,   261,
      -1,   256,    -1,   266,    -1,   268,    -1,   254,   255,    -1,
     267,    -1,   254,    61,   254,    60,    -1,   254,    61,   102,
     254,    60,    -1,   254,    62,     6,    -1,   269,    -1,   255,
      -1,   254,    75,   254,    -1,   254,    75,   254,    77,   270,
      -1,   254,    74,   254,    -1,   254,    73,   254,    -1,   254,
      72,   254,    -1,   254,    71,   254,    -1,   254,    70,   254,
      -1,   254,    64,   254,    -1,   254,    69,   254,    -1,   254,
      68,   254,    -1,   254,    67,   254,    -1,   254,    65,   254,
      -1,   254,    66,   254,    -1,    59,   254,    58,    -1,    61,
      47,    60,    -1,    61,   254,    47,    60,    -1,    61,    47,
     254,    60,    -1,    61,   254,    47,   254,    60,    -1,    61,
     254,    47,   254,    47,   254,    60,    -1,   254,    59,   270,
      58,    -1,   254,    59,    58,    -1,    -1,   254,    59,   270,
       1,   257,    58,    -1,    -1,    48,   259,   260,   198,   136,
       9,    -1,    59,   196,    58,     3,    -1,    59,   196,     1,
      -1,     1,     3,    -1,    -1,    50,   262,   260,   198,   136,
       9,    -1,    -1,    37,   264,   265,    63,   254,    -1,   196,
      -1,     1,     3,    -1,   254,    78,   254,    47,   254,    -1,
     254,    78,   254,    47,     1,    -1,   254,    78,   254,     1,
      -1,   254,    78,     1,    -1,    61,    60,    -1,    61,   270,
      60,    -1,    61,   270,     1,    -1,    52,    60,    -1,    52,
     271,    60,    -1,    52,   271,     1,    -1,    61,    63,    60,
      -1,    61,   273,    60,    -1,    61,   273,     1,    60,    -1,
     254,    -1,   270,    77,   254,    -1,   254,    -1,   271,   254,
      -1,   252,    -1,   272,    77,   252,    -1,   254,    63,   254,
      -1,   273,    77,   254,    63,   254,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   201,   201,   204,   206,   210,   211,   212,   216,   221,
     222,   227,   232,   237,   242,   243,   244,   245,   249,   250,
     254,   260,   266,   274,   275,   276,   277,   278,   279,   284,
     286,   292,   306,   307,   308,   309,   310,   311,   312,   313,
     314,   315,   316,   317,   318,   319,   320,   321,   322,   323,
     324,   328,   334,   342,   344,   349,   349,   363,   371,   372,
     373,   377,   378,   379,   383,   383,   398,   408,   409,   413,
     414,   418,   420,   421,   421,   430,   431,   436,   436,   448,
     449,   452,   454,   460,   469,   477,   487,   496,   506,   505,
     530,   529,   555,   560,   567,   569,   573,   580,   581,   582,
     586,   599,   607,   611,   617,   623,   630,   635,   644,   654,
     654,   668,   677,   681,   681,   694,   703,   707,   707,   723,
     732,   736,   736,   753,   754,   761,   763,   764,   768,   770,
     769,   780,   780,   792,   792,   804,   804,   820,   823,   822,
     835,   836,   837,   840,   841,   847,   848,   852,   861,   873,
     884,   895,   916,   916,   933,   934,   941,   943,   944,   948,
     950,   949,   960,   960,   973,   973,   985,   985,  1003,  1004,
    1007,  1008,  1020,  1041,  1045,  1050,  1058,  1065,  1064,  1083,
    1084,  1087,  1089,  1093,  1094,  1098,  1103,  1121,  1141,  1151,
    1162,  1170,  1171,  1175,  1187,  1210,  1211,  1218,  1228,  1237,
    1238,  1238,  1242,  1246,  1247,  1247,  1254,  1308,  1310,  1311,
    1315,  1330,  1333,  1332,  1344,  1343,  1358,  1359,  1363,  1364,
    1373,  1377,  1385,  1392,  1407,  1413,  1425,  1435,  1440,  1452,
    1461,  1468,  1476,  1481,  1489,  1493,  1501,  1506,  1518,  1523,
    1531,  1532,  1536,  1540,  1552,  1559,  1569,  1570,  1573,  1574,
    1577,  1579,  1583,  1590,  1591,  1592,  1604,  1603,  1662,  1665,
    1671,  1673,  1674,  1674,  1680,  1682,  1686,  1687,  1691,  1725,
    1727,  1736,  1737,  1741,  1742,  1751,  1754,  1756,  1760,  1761,
    1764,  1782,  1786,  1786,  1820,  1842,  1869,  1871,  1872,  1879,
    1887,  1893,  1899,  1913,  1912,  1956,  1958,  1962,  1963,  1968,
    1975,  1975,  1984,  1983,  2047,  2048,  2054,  2056,  2060,  2061,
    2064,  2083,  2092,  2091,  2109,  2110,  2111,  2118,  2134,  2135,
    2136,  2146,  2147,  2148,  2149,  2150,  2151,  2155,  2173,  2174,
    2175,  2186,  2187,  2188,  2189,  2190,  2191,  2192,  2193,  2194,
    2195,  2196,  2197,  2198,  2199,  2200,  2201,  2202,  2203,  2204,
    2205,  2206,  2207,  2208,  2209,  2210,  2211,  2212,  2213,  2214,
    2215,  2216,  2217,  2218,  2219,  2220,  2221,  2222,  2223,  2224,
    2225,  2226,  2227,  2228,  2230,  2235,  2239,  2244,  2250,  2259,
    2260,  2262,  2267,  2274,  2275,  2276,  2277,  2278,  2279,  2280,
    2281,  2282,  2283,  2284,  2285,  2290,  2293,  2296,  2299,  2302,
    2308,  2314,  2319,  2319,  2329,  2328,  2369,  2370,  2374,  2382,
    2381,  2427,  2426,  2469,  2470,  2479,  2484,  2491,  2498,  2508,
    2509,  2513,  2521,  2522,  2526,  2535,  2536,  2537,  2545,  2546,
    2550,  2551,  2555,  2561,  2568,  2569
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "EOL", "INTNUM", "DBLNUM", "SYMBOL",
  "STRING", "NIL", "END", "DEF", "WHILE", "BREAK", "CONTINUE", "DROPPING",
  "IF", "ELSE", "ELIF", "FOR", "FORFIRST", "FORLAST", "FORMIDDLE",
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SENDER", "SELF", "GIVE", "TRY",
  "CATCH", "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL",
  "LAMBDA", "INIT", "LOAD", "LAUNCH", "CONST_KW", "ATTRIBUTES", "PASS",
  "EXPORT", "IMPORT", "DIRECTIVE", "COLON", "FUNCDECL", "STATIC",
  "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "ENUM", "TRUE_TOKEN",
  "FALSE_TOKEN", "OUTER_STRING", "CLOSEPAR", "OPENPAR", "CLOSESQUARE",
  "OPENSQUARE", "DOT", "ARROW", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR",
  "ASSIGN_BXOR", "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV",
  "ASSIGN_MUL", "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_TO", "COMMA",
  "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ", "EEQ",
  "PROVIDES", "OP_NOTIN", "OP_IN", "HASNT", "HAS", "DIESIS", "ATSIGN",
  "CAP", "VBAR", "AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "BANG", "NEG", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement", "base_statement",
  "assignment_def_list", "def_statement", "while_statement", "@1",
  "while_decl", "while_short_decl", "if_statement", "@2", "if_decl",
  "if_short_decl", "elif_or_else", "@3", "else_decl", "elif_statement",
  "@4", "elif_decl", "statement_list", "break_statement",
  "continue_statement", "forin_statement", "@5", "@6",
  "forin_statement_list", "forin_statement_elem", "fordot_statement",
  "self_print_statement", "outer_print_statement", "first_loop_block",
  "@7", "last_loop_block", "@8", "middle_loop_block", "@9",
  "switch_statement", "@10", "switch_decl", "case_list", "case_statement",
  "@11", "@12", "@13", "@14", "default_statement", "@15", "default_decl",
  "default_body", "case_expression_list", "case_element",
  "select_statement", "@16", "select_decl", "selcase_list",
  "selcase_statement", "@17", "@18", "@19", "@20",
  "selcase_expression_list", "selcase_element", "give_statement",
  "try_statement", "@21", "try_decl", "catch_statements", "catch_list",
  "catch_statement", "catch_decl", "catchcase_element_list",
  "catchcase_element", "raise_statement", "func_statement", "func_decl",
  "@22", "func_decl_short", "@23", "func_begin", "param_list",
  "param_symbol", "static_block", "@24", "@25", "static_decl",
  "static_short_decl", "launch_statement", "pass_statement",
  "const_statement", "export_statement", "export_symbol_list",
  "import_statement", "import_symbol_list", "directive_statement",
  "directive_pair_list", "directive_pair", "attributes_statement",
  "attributes_decl", "attributes_short_decl", "attribute_list",
  "attribute_vert_list", "class_decl", "@26", "class_def_inner",
  "class_param_list", "@27", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "inherit_param_list",
  "inherit_param_token", "class_statement_list", "class_statement",
  "init_decl", "@28", "property_decl", "has_list", "has_clause_list",
  "enum_statement", "@29", "enum_statement_list", "enum_item_decl",
  "enum_item_terminator", "object_decl", "@30", "object_decl_inner",
  "object_statement_list", "object_statement", "global_statement", "@31",
  "global_symbol_list", "globalized_symbol", "return_statement",
  "const_atom", "atomic_symbol", "var_atom", "expression", "range_decl",
  "func_call", "@32", "nameless_func", "@33", "nameless_func_decl_inner",
  "nameless_closure", "@34", "lambda_expr", "@35", "lambda_expr_inner",
  "iif_expr", "array_decl", "dotarray_decl", "dict_decl",
  "expression_list", "listpar_expression_list", "symbol_list",
  "expression_pair_list", 0
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
     365
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   111,   112,   113,   113,   114,   114,   114,   115,   115,
     115,   115,   115,   115,   115,   115,   115,   115,   116,   116,
     117,   117,   117,   118,   118,   118,   118,   118,   118,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   119,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   119,   119,
     119,   120,   120,   121,   121,   123,   122,   122,   124,   124,
     124,   125,   125,   125,   127,   126,   126,   128,   128,   129,
     129,   130,   130,   131,   130,   132,   132,   134,   133,   135,
     135,   136,   136,   137,   137,   138,   138,   138,   140,   139,
     141,   139,   139,   139,   142,   142,   143,   143,   143,   143,
     144,   144,   145,   145,   145,   145,   145,   145,   146,   148,
     147,   147,   147,   150,   149,   149,   149,   152,   151,   151,
     151,   154,   153,   155,   155,   156,   156,   156,   157,   158,
     157,   159,   157,   160,   157,   161,   157,   162,   163,   162,
     164,   164,   164,   165,   165,   166,   166,   167,   167,   167,
     167,   167,   169,   168,   170,   170,   171,   171,   171,   172,
     173,   172,   174,   172,   175,   172,   176,   172,   177,   177,
     178,   178,   178,   179,   179,   179,   180,   181,   180,   182,
     182,   183,   183,   184,   184,   185,   186,   186,   186,   186,
     186,   187,   187,   188,   188,   189,   189,   190,   190,   191,
     192,   191,   191,   193,   194,   193,   195,   196,   196,   196,
     197,   198,   199,   198,   200,   198,   201,   201,   202,   202,
     203,   203,   204,   204,   204,   204,   205,   205,   205,   206,
     206,   206,   207,   207,   208,   208,   209,   209,   210,   210,
     211,   211,   212,   212,   213,   213,   214,   214,   215,   215,
     216,   216,   216,   217,   217,   217,   219,   218,   220,   220,
     221,   221,   222,   221,   223,   223,   224,   224,   225,   226,
     226,   227,   227,   228,   228,   228,   229,   229,   230,   230,
     230,   230,   232,   231,   233,   233,   234,   234,   234,   235,
     235,   235,   235,   237,   236,   238,   238,   239,   239,   239,
     240,   240,   242,   241,   243,   243,   244,   244,   245,   245,
     245,   245,   247,   246,   248,   248,   248,   249,   250,   250,
     250,   251,   251,   251,   251,   251,   251,   252,   253,   253,
     253,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   255,   255,   255,   255,   255,
     256,   256,   257,   256,   259,   258,   260,   260,   260,   262,
     261,   264,   263,   265,   265,   266,   266,   266,   266,   267,
     267,   267,   268,   268,   268,   269,   269,   269,   270,   270,
     271,   271,   272,   272,   273,   273
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     3,     3,     3,     2,     2,
       4,     6,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     1,
       3,     3,     2,     3,     0,     6,     2,     3,     3,     3,
       3,     0,     1,     0,     3,     2,     3,     0,     4,     3,
       3,     0,     2,     2,     3,     2,     3,     3,     0,     9,
       0,     7,     5,     3,     0,     2,     1,     1,     1,     1,
       3,     3,     3,     2,     3,     2,     3,     3,     1,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     0,     0,     3,
       0,     1,     2,     2,     2,     1,     3,     1,     1,     1,
       3,     1,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     1,     3,
       0,     1,     1,     5,     4,     3,     3,     0,     6,     2,
       3,     0,     1,     1,     2,     2,     2,     4,     3,     5,
       3,     1,     3,     1,     1,     3,     3,     5,     2,     5,
       0,     7,     3,     5,     0,     6,     2,     0,     1,     3,
       1,     0,     0,     5,     0,     3,     2,     3,     2,     3,
       3,     3,     3,     5,     5,     3,     5,     5,     3,     2,
       3,     3,     1,     3,     3,     3,     1,     3,     3,     3,
       1,     3,     3,     3,     4,     3,     2,     3,     2,     3,
       0,     1,     3,     2,     3,     2,     0,     8,     3,     2,
       0,     3,     0,     5,     0,     2,     1,     3,     2,     0,
       3,     1,     3,     1,     1,     1,     0,     2,     1,     1,
       1,     1,     0,     7,     5,     4,     0,     3,     3,     1,
       2,     3,     4,     0,     7,     0,     2,     1,     4,     2,
       1,     1,     0,     8,     2,     2,     0,     2,     1,     1,
       1,     1,     0,     4,     1,     3,     3,     1,     2,     3,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     2,     2,     2,
       2,     3,     3,     3,     3,     3,     3,     3,     2,     3,
       3,     3,     3,     3,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     2,     1,     4,     5,     3,     1,
       1,     3,     5,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     4,     4,     5,     7,
       4,     3,     0,     6,     0,     6,     4,     3,     2,     0,
       6,     0,     5,     1,     2,     5,     5,     4,     3,     2,
       3,     3,     2,     3,     3,     3,     3,     4,     1,     3,
       1,     2,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   324,   325,   327,   326,
     321,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   330,   329,     0,     0,     0,     0,     0,     0,   312,
     411,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     409,     0,     0,    59,     0,   322,   323,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       4,     5,     8,    13,    23,    32,    33,    55,     0,    37,
      64,     0,    38,    39,    34,    47,    48,    49,    35,   121,
      36,   152,    40,    41,   177,    42,    10,   211,     0,     0,
      45,    46,    14,    15,    16,     9,    17,     0,   250,    11,
      50,    12,    44,    43,   331,   328,   332,   428,   380,   371,
     369,   370,   368,   372,   375,   373,   379,     0,    28,     6,
       0,     0,     0,     0,   404,     0,     0,    83,     0,    85,
       0,     0,     0,     0,   432,     0,     0,     0,     0,     0,
       0,     0,   428,     0,     0,   179,     0,     0,     0,     0,
     256,     0,   302,     0,   318,     0,     0,     0,     0,     0,
       0,     0,     0,   371,     0,     0,     0,   246,   248,     0,
       0,     0,   229,   232,     0,     0,   232,     0,     0,     0,
       0,     0,   240,     0,   206,     0,     0,     0,     0,   422,
     430,     0,    62,   293,     0,     0,   419,     0,   428,     0,
       0,   358,     0,   105,     0,   367,   366,   333,     0,   103,
       0,   345,   350,   348,   365,   364,    81,     0,     0,     0,
      57,    81,    66,   125,   156,    81,     0,    81,   212,   214,
     198,     0,     0,     0,   251,     0,   250,     0,    29,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   349,
     347,   374,     0,     0,    54,    53,     0,     0,    60,    63,
      58,    61,    84,    87,    86,    68,    70,    67,    69,    93,
       0,     0,   124,   123,     7,   155,   154,   175,     0,     0,
     180,   176,   196,   195,    27,     0,    26,     0,   320,   319,
     317,     0,   314,     0,   210,   413,   208,     0,    22,    20,
      21,   221,   220,   228,     0,   247,   249,   225,   222,     0,
     231,   230,     0,   235,     0,   234,   239,     0,   238,     0,
      25,     0,   207,   211,   211,   101,   100,   424,   423,   431,
       0,   394,   395,     0,   425,     0,     0,   421,   420,     0,
     426,     0,   107,   104,   106,   102,     0,     0,     0,     0,
       0,     0,   216,   218,     0,    81,     0,   202,   204,     0,
     255,   253,     0,     0,     0,   245,   401,     0,     0,     0,
     378,   388,   392,   393,   391,   390,   389,   387,   386,   385,
     384,   383,   381,   418,     0,   357,   356,   355,   354,   353,
     352,   346,   351,   363,   362,   361,   360,   359,   342,   341,
     340,   335,   334,   338,   337,   336,   339,   344,   343,     0,
     429,     0,    51,   433,     0,     0,   174,     0,     0,   207,
     276,   264,     0,     0,     0,   306,   313,     0,   414,     0,
       0,     0,     0,     0,   361,   233,   233,    18,   242,     0,
     243,   241,   408,     0,    81,    81,   295,   397,   396,     0,
     434,   427,     0,     0,    82,     0,     0,     0,    73,    72,
      77,     0,   128,     0,     0,   126,     0,   138,     0,   159,
       0,     0,   157,     0,     0,   182,   183,    81,   217,   219,
       0,     0,   215,     0,   200,     0,   252,   244,   254,   402,
     400,     0,   376,     0,   417,     0,    30,     0,     0,    92,
      88,    90,   173,   259,     0,   286,     0,   305,   269,   265,
     266,   304,   286,   316,   315,   209,   412,   227,   226,   224,
     223,    19,   407,     0,     0,     0,     0,     0,   398,     0,
      56,     0,    75,     0,     0,     0,    81,    81,   127,     0,
     151,   149,   147,   148,     0,   145,   142,     0,     0,   158,
       0,   171,   172,     0,   168,     0,     0,   186,   193,   194,
       0,     0,   191,     0,   184,     0,   197,     0,     0,     0,
     199,   203,     0,   377,   382,   416,   415,     0,    52,     0,
       0,   262,   261,   278,     0,     0,     0,     0,     0,   279,
     277,   281,   280,     0,   258,     0,   268,     0,   308,   309,
     311,   310,     0,   307,   406,   405,   410,   297,     0,     0,
     296,     0,   435,    76,    80,    79,    65,     0,     0,   133,
     135,     0,   129,   131,     0,   122,    81,     0,   139,   164,
     166,   160,   162,   170,   153,   190,     0,   188,     0,     0,
     178,   213,   205,     0,   403,    31,     0,     0,     0,    96,
       0,     0,    97,    98,    99,    91,     0,     0,   282,     0,
       0,   289,     0,     0,     0,   274,   275,     0,   271,   273,
     267,     0,   300,     0,   301,   299,   294,   399,    78,    81,
       0,   150,    81,     0,   146,     0,   144,    81,     0,    81,
       0,   169,   187,   192,     0,   201,     0,   109,     0,     0,
     113,     0,     0,   117,     0,     0,    95,   263,     0,   211,
       0,   288,   290,   287,     0,   257,   270,     0,   303,     0,
       0,   136,     0,   132,     0,   167,     0,   163,   189,   112,
      81,   111,   116,    81,   115,   120,    81,   119,    89,   285,
      81,     0,   291,     0,   272,   298,     0,     0,     0,     0,
     284,   292,     0,     0,     0,     0,   110,   114,   118,   283
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    60,    61,   573,    62,   484,    64,   121,
      65,    66,   216,    67,    68,    69,   221,    70,    71,   487,
     566,   488,   489,   567,   490,   376,    72,    73,    74,   609,
     610,   680,   681,    75,    76,    77,   682,   760,   683,   763,
     684,   766,    78,   223,    79,   378,   495,   712,   713,   709,
     710,   496,   578,   497,   658,   574,   575,    80,   224,    81,
     379,   502,   719,   720,   717,   718,   583,   584,    82,    83,
     225,    84,   504,   505,   506,   507,   591,   592,    85,    86,
      87,   599,    88,   513,    89,   325,   326,   227,   385,   386,
     228,   229,    90,    91,    92,    93,   174,    94,   178,    95,
     181,   182,    96,    97,    98,   235,   236,    99,   315,   450,
     451,   686,   454,   539,   540,   626,   697,   698,   535,   620,
     621,   739,   622,   623,   693,   100,   360,   556,   640,   705,
     101,   317,   455,   542,   633,   102,   156,   321,   322,   103,
     104,   105,   106,   107,   108,   109,   602,   110,   185,   353,
     111,   186,   112,   157,   327,   113,   114,   115,   116,   117,
     191,   135,   200
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -345
static const yytype_int16 yypact[] =
{
    -345,    72,   755,  -345,    85,  -345,  -345,  -345,  -345,  -345,
    -345,    98,   184,  3207,    57,   325,  3270,   277,  3333,   112,
    3396,  -345,  -345,  3459,    49,  3522,   331,   350,   475,  -345,
    -345,   340,  3585,   351,   301,  3648,   452,   463,   466,   271,
    -345,  3711,  5057,   117,   152,  -345,  -345,  -345,  5309,  4930,
    5309,  3018,  5309,  5309,  5309,  3081,  5309,  5309,  5309,     4,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  2955,  -345,
    -345,  2955,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,    75,  2955,    74,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,    77,   167,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  4360,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,   368,  -345,  -345,
     186,    78,   148,   114,  -345,  4138,   295,  -345,   313,  -345,
     320,   135,  4198,   326,  -345,   196,   341,  4412,   355,   359,
    4463,   375,  5786,    55,   376,  -345,  2955,   396,  4515,   406,
    -345,   427,  -345,   433,  -345,  4566,   440,    27,   444,   451,
     457,   460,  5786,   467,   483,   412,   293,  -345,  -345,   493,
    4618,   494,  -345,  -345,    82,   497,   503,   445,   507,   511,
     443,    95,  -345,   530,  -345,    81,    81,   538,  4669,  -345,
    5786,  3144,  -345,  -345,  5531,  5120,  -345,   477,  5361,    86,
      93,  6041,   539,  -345,   109,   515,   515,   256,   540,  -345,
     124,   256,   256,   256,  -345,  -345,  -345,   543,   545,   101,
    -345,  -345,  -345,  -345,  -345,  -345,   319,  -345,  -345,  -345,
    -345,   548,    26,   549,  -345,   133,   428,   191,  -345,  5183,
    4994,   547,  5309,  5309,  5309,  5309,  5309,  5309,  5309,  5309,
    5309,  5309,  5309,  5309,  3774,  5309,  5309,  5309,  5309,  5309,
    5309,  5309,  5309,   551,  5309,  5309,  5309,  5309,  5309,  5309,
    5309,  5309,  5309,  5309,  5309,  5309,  5309,  5309,  5309,  -345,
    -345,  -345,  5309,  5309,  -345,  -345,   552,  5309,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
     552,  3837,  -345,  -345,  -345,  -345,  -345,  -345,   556,  5309,
    -345,  -345,  -345,  -345,  -345,   266,  -345,   134,  -345,  -345,
    -345,   197,  -345,   557,  -345,   478,  -345,   491,  -345,  -345,
    -345,  -345,  -345,  -345,   280,  -345,  -345,  -345,  -345,  3900,
    -345,  -345,   555,  -345,   558,  -345,  -345,    47,  -345,   559,
    -345,   567,   565,    75,    75,  -345,  -345,  -345,  -345,  5786,
     569,  -345,  -345,  5582,  -345,  5246,  5309,  -345,  -345,   518,
    -345,  5309,  -345,  -345,  -345,  -345,  1745,  1415,   653,   657,
    1525,   303,  -345,  -345,  1855,  -345,  2955,  -345,  -345,    33,
    -345,  -345,   573,   572,   202,  -345,  -345,    91,  5309,  5420,
    -345,  5786,  5786,  5786,  5786,  5786,  5786,  5786,  5786,  5786,
    5786,  5786,  5837,  -345,  4078,  5990,  6041,   544,   544,   544,
     544,   544,   544,  -345,   515,   515,   515,   515,   272,   272,
    4968,   390,   390,   400,   400,   400,   458,   256,   256,  4249,
    5888,   512,  5786,  -345,   579,  4309,  -345,   203,   583,   565,
    -345,   560,   585,   584,   586,  -345,  -345,   523,  -345,   565,
    5309,   588,   592,   593,   116,  -345,   594,  -345,  -345,   595,
    -345,  -345,  -345,    94,  -345,  -345,  -345,  -345,  -345,  5479,
    5786,  -345,  5633,   597,  -345,   470,  3963,   589,  -345,  -345,
    -345,   598,  -345,    11,   437,  -345,   600,  -345,   599,  -345,
      73,   612,  -345,    67,   613,   574,  -345,  -345,  -345,  -345,
     622,  1965,  -345,   568,  -345,   308,  -345,  -345,  -345,  -345,
    -345,  5684,  -345,  5309,  -345,  4026,  -345,  5309,  5309,  -345,
    -345,  -345,  -345,  -345,   104,    51,   626,  -345,   571,   554,
    -345,  -345,    59,  -345,  -345,  -345,  5786,  -345,  -345,  -345,
    -345,  -345,  -345,   634,  2075,  2185,   426,  5309,  -345,  5309,
    -345,   647,  -345,   648,  4721,   654,  -345,  -345,  -345,   316,
    -345,  -345,  -345,   587,   118,  -345,  -345,   658,   321,  -345,
     335,  -345,  -345,   119,  -345,   661,   662,  -345,  -345,  -345,
     552,    52,  -345,   664,  -345,  1635,  -345,   665,   623,   611,
    -345,  -345,   614,  -345,   596,  -345,  5939,   205,  5786,   865,
    2955,  -345,  -345,  -345,   603,   668,   676,   677,    58,  -345,
    -345,  -345,  -345,   675,  -345,   287,  -345,   584,  -345,  -345,
    -345,  -345,   678,  -345,  -345,  -345,  -345,  -345,   127,   671,
    -345,  5735,  5786,  -345,  -345,  -345,  -345,  2295,  1415,  -345,
    -345,    17,  -345,  -345,    18,  -345,  -345,  2955,  -345,  -345,
    -345,  -345,  -345,   522,  -345,  -345,   682,  -345,   534,   552,
    -345,  -345,  -345,   683,  -345,  -345,   438,   441,   474,  -345,
     679,   865,  -345,  -345,  -345,  -345,   631,  5309,  -345,   615,
     689,  -345,   687,   223,   692,  -345,  -345,   111,  -345,  -345,
    -345,   693,  -345,   305,  -345,  -345,  -345,  -345,  -345,  -345,
    2955,  -345,  -345,  2955,  -345,  2405,  -345,  -345,  2955,  -345,
    2955,  -345,  -345,  -345,   694,  -345,   695,  -345,  2955,   699,
    -345,  2955,   700,  -345,  2955,   702,  -345,  -345,  4772,    75,
    5309,  -345,  -345,  -345,    25,  -345,  -345,   287,  -345,   226,
     975,  -345,  1085,  -345,  1195,  -345,  1305,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  4824,  -345,   701,  -345,  -345,  2515,  2625,  2735,  2845,
    -345,  -345,   703,   706,   707,   709,  -345,  -345,  -345,  -345
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -345,  -345,  -345,  -345,  -345,  -344,  -345,    -2,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,    65,
    -345,  -345,  -345,  -345,  -345,   -18,  -345,  -345,  -345,  -345,
    -345,    34,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,
    -345,   338,  -345,  -345,  -345,  -345,    60,  -345,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,    56,  -345,  -345,
    -345,  -345,  -345,  -345,   215,  -345,  -345,    53,  -345,  -234,
    -345,  -345,  -345,  -345,  -345,  -227,   263,  -306,  -345,  -345,
    -345,  -345,  -345,  -345,  -345,  -345,   686,  -345,  -345,  -345,
    -345,   378,  -345,  -345,  -345,   -89,  -345,  -345,  -345,  -345,
    -345,  -345,   274,  -345,   103,  -345,  -345,   -16,  -345,  -345,
     192,  -345,   193,   194,  -345,  -345,  -345,  -345,  -345,   -11,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,   276,  -345,
    -320,   -10,  -345,   -12,   -14,   705,  -345,  -345,  -345,   553,
    -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,  -345,    12,
    -345,  -345,  -345
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -405
static const yytype_int16 yytable[] =
{
      63,   125,   122,   470,   132,   389,   137,   134,   140,   237,
       8,   142,   569,   148,   462,   467,   155,   570,   571,   572,
     162,   467,   467,   170,   570,   571,   572,   388,   323,   188,
     190,   772,   324,   324,   514,   143,   194,   198,   201,   142,
     205,   206,   207,   142,   211,   212,   213,   474,   475,   215,
     144,   467,   145,   468,   613,   667,   308,   614,   126,   690,
     127,   199,   628,   204,   691,   614,   220,   210,   586,   222,
     587,   588,     3,   589,   580,   231,  -170,   581,   233,   582,
    -250,   285,   351,   234,  -207,   341,   230,   367,   118,   615,
    -207,   515,   519,   281,   369,   552,   146,   615,   348,   616,
     617,   119,   183,  -207,  -207,   611,   773,   616,   617,   469,
     459,   281,   373,   138,   214,   469,   469,   288,   281,   550,
    -170,   652,   661,   281,   226,   473,   281,   375,   281,   668,
     702,   309,   283,   232,   281,   452,   391,  -264,   295,   692,
     352,   281,   669,   618,   311,   469,   368,   394,   281,   520,
    -170,   618,   553,   370,  -250,   286,   281,   590,   193,   342,
    -404,   289,   612,   283,   192,   653,   662,   453,   283,   746,
     371,   459,   349,   234,   281,   239,   281,   240,   241,   359,
     281,   459,   296,   363,   281,   120,   283,   281,   747,   284,
       8,   281,   281,   281,   395,   654,   663,   281,   281,   281,
     456,   283,   703,   377,   704,   518,   532,   380,   675,   384,
     392,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   534,   287,   279,   280,   743,   142,   399,   702,
     401,   402,   403,   404,   405,   406,   407,   408,   409,   410,
     411,   412,   414,   415,   416,   417,   418,   419,   420,   421,
     422,   397,   424,   425,   426,   427,   428,   429,   430,   431,
     432,   433,   434,   435,   436,   437,   438,   448,   392,  -260,
     439,   440,   183,   300,   457,   442,   441,   184,   133,   392,
     283,   461,   283,     8,     6,     7,   301,     9,    10,   445,
     443,     6,     7,   695,     9,    10,   335,   142,   292,  -260,
     744,   619,   166,   704,   167,   699,   508,   711,   629,     6,
       7,   600,     9,    10,   696,   239,   293,   240,   241,   649,
     381,   447,   382,   294,   656,   449,   128,   464,   129,   299,
    -404,   239,   149,   240,   241,    45,    46,   150,   659,   130,
     336,   158,    45,    46,   302,   281,   159,   160,   168,   281,
     509,   151,   164,   479,   480,   601,   152,   165,   304,   482,
      45,    46,   305,   650,   279,   280,   383,   511,   657,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   307,   310,
     279,   280,   660,   749,   512,   281,   521,   281,   281,   281,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   312,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   314,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   281,   281,   281,   699,   281,   637,
     316,   281,   638,   770,   234,   639,   318,   393,   576,   726,
    -141,   727,   729,   282,   730,   283,   320,   328,   546,   239,
     281,   240,   241,   171,   329,   172,   554,   555,   173,   239,
     330,   240,   241,   331,   175,   281,   281,   179,   281,   176,
     332,   561,   180,   562,   564,   732,   153,   733,   154,     6,
       7,     8,     9,    10,  -141,   728,   333,   334,   731,   595,
     273,   274,   275,   276,   277,   278,   337,   340,   279,   280,
     343,    21,    22,   276,   277,   278,  -236,   281,   279,   280,
     345,   142,    30,   606,   346,   142,   608,   239,   347,   240,
     241,   734,   344,   124,   543,    40,   581,    42,   582,   320,
      45,    46,   281,   350,    48,   604,    49,   364,   588,   607,
     589,   355,   372,   374,   149,   641,   151,   642,   647,   648,
     281,   387,   390,   400,   460,   459,    50,   423,     8,   446,
     458,   465,   277,   278,   466,   180,   279,   280,    52,    53,
     472,   324,   476,    54,   239,   517,   240,   241,   481,   516,
     666,    56,   529,    57,    58,    59,   533,   528,   537,   541,
     538,   547,   281,   453,   281,   548,   549,  -237,   565,   551,
     560,   568,   579,   239,   503,   240,   241,   679,   685,   577,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   585,   593,   279,   280,   596,   598,   281,   281,   624,
     625,   627,   263,   264,   265,   266,   267,   634,   715,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     643,   644,   279,   280,   491,   716,   492,   646,   498,   724,
     499,   655,  -137,   651,   664,   665,  -137,   670,   671,   673,
     672,   688,   674,   283,   706,   738,   493,   494,   687,   679,
     500,   494,   184,   689,   694,   722,   725,   701,   735,   737,
     740,   750,   741,   742,   752,   745,   748,   758,   759,   754,
    -140,   756,   762,   765,  -140,   768,   786,   781,   751,   787,
     788,   753,   789,   708,   714,   736,   755,   501,   757,   721,
     594,   723,   545,   177,   281,   536,   761,   471,   771,   764,
     700,   774,   767,   544,   630,   631,   632,   163,   775,   354,
       0,     0,   776,     0,     0,   777,     0,     0,   778,     0,
       0,     0,   779,     0,     0,    -2,     4,   281,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,    19,     0,
      20,    21,    22,    23,    24,     0,    25,    26,     0,    27,
      28,    29,    30,     0,    31,    32,    33,    34,    35,    36,
      37,    38,     0,    39,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   -94,    12,    13,    14,    15,     0,
      16,     0,     0,    17,   676,   677,   678,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -134,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -134,  -134,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -134,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -130,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -130,  -130,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -130,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -165,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -165,  -165,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -165,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -161,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -161,  -161,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,  -161,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   -71,    12,    13,    14,    15,     0,
      16,   485,   486,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -181,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,   503,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -185,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,  -185,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   483,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   510,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   597,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   635,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   636,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   -74,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,  -143,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   782,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   783,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   784,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   785,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,   202,
       0,   203,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,    21,    22,     0,     0,    52,    53,
       0,     0,     0,    54,     0,    30,     0,     0,     0,    55,
       0,    56,     0,    57,    58,    59,   124,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,   208,     0,   209,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,    52,    53,     0,     0,     0,    54,     0,    30,     0,
       0,     0,     0,     0,    56,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,   357,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,    52,    53,     0,     0,     0,    54,
       0,    30,     0,     0,     0,     0,     0,    56,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,   358,    49,     0,     0,   123,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    52,    53,     0,
       0,     0,    54,     0,    30,     0,     0,     0,     0,     0,
      56,     0,    57,    58,    59,   124,     0,    40,     0,    42,
       0,     0,    45,    46,     0,     0,    48,     0,    49,     0,
       0,   131,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
      52,    53,     0,     0,     0,    54,     0,    30,     0,     0,
       0,     0,     0,    56,     0,    57,    58,    59,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,   136,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    52,    53,     0,     0,     0,    54,     0,
      30,     0,     0,     0,     0,     0,    56,     0,    57,    58,
      59,   124,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,     0,    49,     0,     0,   139,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    52,    53,     0,     0,
       0,    54,     0,    30,     0,     0,     0,     0,     0,    56,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
     141,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,    52,
      53,     0,     0,     0,    54,     0,    30,     0,     0,     0,
       0,     0,    56,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,   147,     0,     0,     6,     7,     8,     9,
      10,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,    21,    22,
       0,     0,    52,    53,     0,     0,     0,    54,     0,    30,
       0,     0,     0,     0,     0,    56,     0,    57,    58,    59,
     124,     0,    40,     0,    42,     0,     0,    45,    46,     0,
       0,    48,     0,    49,     0,     0,   161,     0,     0,     6,
       7,     8,     9,    10,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    50,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,    52,    53,     0,     0,     0,
      54,     0,    30,     0,     0,     0,     0,     0,    56,     0,
      57,    58,    59,   124,     0,    40,     0,    42,     0,     0,
      45,    46,     0,     0,    48,     0,    49,     0,     0,   169,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
       0,     0,     0,     0,    21,    22,     0,     0,    52,    53,
       0,     0,     0,    54,     0,    30,     0,     0,     0,     0,
       0,    56,     0,    57,    58,    59,   124,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,   187,     0,     0,     6,     7,     8,     9,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,    21,    22,     0,
       0,    52,    53,     0,     0,     0,    54,     0,    30,     0,
       0,     0,     0,     0,    56,     0,    57,    58,    59,   124,
       0,    40,     0,    42,     0,     0,    45,    46,     0,     0,
      48,     0,    49,     0,     0,   413,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,     0,    52,    53,     0,     0,     0,    54,
       0,    30,     0,     0,     0,     0,     0,    56,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,   444,     0,
       0,     6,     7,     8,     9,    10,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    52,    53,     0,
       0,     0,    54,     0,    30,     0,     0,     0,     0,     0,
      56,     0,    57,    58,    59,   124,     0,    40,     0,    42,
       0,     0,    45,    46,     0,     0,    48,     0,    49,     0,
       0,   463,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
      52,    53,     0,     0,     0,    54,     0,    30,     0,     0,
       0,     0,     0,    56,     0,    57,    58,    59,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
       0,    49,     0,     0,   563,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    52,    53,     0,     0,     0,    54,     0,
      30,     0,     0,     0,     0,     0,    56,     0,    57,    58,
      59,   124,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,     0,    49,     0,     0,   605,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    52,    53,     0,     0,
       0,    54,     0,    30,     0,     0,     0,     0,     0,    56,
       0,    57,    58,    59,   124,     0,    40,     0,    42,   524,
       0,    45,    46,     0,     0,    48,     0,    49,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,   525,     0,     0,     0,     0,
       0,     0,    56,     0,    57,    58,    59,   239,     0,   240,
     241,   290,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,   291,   279,   280,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   239,     0,   240,
     241,   297,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,   298,   279,   280,     0,     0,
       0,     0,   526,     0,     0,     0,     0,   239,     0,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,   279,   280,   239,     0,
     240,   241,   530,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,   527,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,   531,   279,   280,     0,
       0,     0,     0,   238,     0,     0,     0,     0,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   303,     0,   279,   280,   239,
       0,   240,   241,     0,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   306,     0,   279,   280,
       0,   239,     0,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   313,     0,
     279,   280,   239,     0,   240,   241,     0,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   319,
       0,   279,   280,     0,   239,     0,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   338,     0,   279,   280,   239,     0,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   356,     0,   279,   280,     0,   239,     0,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   339,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   645,     0,   279,   280,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   769,     0,   279,   280,     0,
     239,     0,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   780,     0,   279,
     280,   239,     0,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
     279,   280,     0,   239,     0,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,   279,   280,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    30,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   195,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
     196,    49,     0,   197,     0,     0,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,     0,    52,    53,     0,     0,   239,    54,   240,
     241,    30,     0,     0,     0,     0,    56,     0,    57,    58,
      59,   195,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
       0,     6,     7,     8,     9,    10,   271,   272,   273,   274,
     275,   276,   277,   278,     0,    50,   279,   280,     0,     0,
       0,     0,     0,    21,    22,     0,     0,    52,    53,     0,
       0,     0,    54,     0,    30,     0,   398,     0,     0,     0,
      56,     0,    57,    58,    59,   124,     0,    40,     0,    42,
       0,     0,    45,    46,     0,     0,    48,   189,    49,     0,
       0,     0,     0,     0,     6,     7,     8,     9,    10,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,    21,    22,     0,     0,
      52,    53,     0,     0,     0,    54,     0,    30,     0,     0,
       0,     0,     0,    56,     0,    57,    58,    59,   124,     0,
      40,     0,    42,     0,     0,    45,    46,     0,     0,    48,
     362,    49,     0,     0,     0,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,    52,    53,     0,     0,     0,    54,     0,
      30,     0,     0,     0,     0,     0,    56,     0,    57,    58,
      59,   124,     0,    40,     0,    42,     0,     0,    45,    46,
       0,   396,    48,     0,    49,     0,     0,     0,     0,     0,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,    52,    53,     0,     0,
       0,    54,     0,    30,     0,     0,     0,     0,     0,    56,
       0,    57,    58,    59,   124,     0,    40,     0,    42,     0,
       0,    45,    46,     0,     0,    48,   478,    49,     0,     0,
       0,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    50,     0,     0,
       0,     0,     0,     0,     0,    21,    22,     0,     0,    52,
      53,     0,     0,     0,    54,     0,    30,     0,     0,     0,
       0,     0,    56,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,   365,     0,
       0,     0,     0,     0,     0,    56,     0,    57,    58,    59,
     239,     0,   240,   241,   366,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   365,     0,   279,
     280,     0,     0,     0,     0,     0,     0,     0,     0,   239,
     522,   240,   241,     0,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   557,     0,   279,   280,
       0,     0,     0,     0,     0,     0,     0,     0,   239,   558,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,   279,   280,   361,
     239,     0,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280,   239,   477,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
     279,   280,   239,     0,   240,   241,   559,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,   279,   280,   239,   603,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,   279,   280,   239,   707,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,   279,   280,   239,     0,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,   279,   280,   239,     0,   240,   241,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   253,     0,   523,   254,   255,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,   279,   280,   239,     0,   240,
     241,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,   279,   280,   239,     0,
     240,   241,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,   279,   280,   239,
       0,   240,   241,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,   279,   280,
     239,     0,   240,   241,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,   279,
     280
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   347,    16,   232,    18,    17,    20,    98,
       6,    23,     1,    25,   334,     4,    28,     6,     7,     8,
      32,     4,     4,    35,     6,     7,     8,     1,     1,    41,
      42,     6,     6,     6,     1,    23,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,   353,   354,    59,
       1,     4,     3,     6,     3,     3,     1,     6,     1,     1,
       3,    49,     3,    51,     6,     6,    68,    55,     1,    71,
       3,     4,     0,     6,     1,     1,     3,     4,     1,     6,
       3,     3,     1,     6,    58,     3,    88,     1,     3,    38,
      63,    58,     1,   107,     1,     1,    47,    38,     3,    48,
      49,     3,     1,    77,    77,     1,    81,    48,    49,    98,
      77,   125,     3,     1,   110,    98,    98,     3,   132,     3,
      47,     3,     3,   137,    49,   352,   140,     3,   142,    77,
       3,    76,    77,    59,   148,     1,     3,     3,     3,    81,
      59,   155,    90,    92,   146,    98,    60,   236,   162,    58,
      77,    92,    58,    60,    77,    77,   170,    90,     6,    77,
      59,    47,    58,    77,    47,    47,    47,    33,    77,    58,
      77,    77,    77,     6,   188,    59,   190,    61,    62,   191,
     194,    77,    47,   195,   198,     1,    77,   201,    77,     3,
       6,   205,   206,   207,     3,    77,    77,   211,   212,   213,
       3,    77,    75,   221,    77,     3,     3,   225,     3,   227,
      77,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   449,    75,   108,   109,     3,   239,   240,     3,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   239,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     1,    77,     3,
     282,   283,     1,    77,    77,   287,   286,     6,     1,    77,
      77,     1,    77,     6,     4,     5,    90,     7,     8,   301,
     300,     4,     5,     6,     7,     8,     3,   309,     3,    33,
      77,   535,     1,    77,     3,   625,     3,   651,   542,     4,
       5,     3,     7,     8,    27,    59,     3,    61,    62,     3,
       1,   309,     3,     3,     3,    59,     1,   339,     3,     3,
      59,    59,     1,    61,    62,    55,    56,     6,     3,    14,
      47,     1,    55,    56,     3,   359,     6,     7,    47,   363,
      47,     1,     1,   365,   366,    47,     6,     6,     3,   371,
      55,    56,     3,    47,   108,   109,    47,   385,    47,    97,
      98,    99,   100,   101,   102,   103,   104,   105,     3,     3,
     108,   109,    47,   703,   386,   399,   398,   401,   402,   403,
     404,   405,   406,   407,   408,   409,   410,   411,   412,     3,
     414,   415,   416,   417,   418,   419,   420,   421,   422,     3,
     424,   425,   426,   427,   428,   429,   430,   431,   432,   433,
     434,   435,   436,   437,   438,   439,   440,   747,   442,     3,
       3,   445,     6,   739,     6,     9,     3,     9,     1,     1,
       3,     3,     1,    75,     3,    77,     6,     3,   460,    59,
     464,    61,    62,     1,     3,     3,   474,   475,     6,    59,
       3,    61,    62,     3,     1,   479,   480,     1,   482,     6,
       3,     1,     6,     3,   486,     1,     1,     3,     3,     4,
       5,     6,     7,     8,    47,    47,     3,    75,    47,   507,
     100,   101,   102,   103,   104,   105,     3,     3,   108,   109,
       3,    26,    27,   103,   104,   105,     3,   521,   108,   109,
       3,   523,    37,   525,     3,   527,   528,    59,    75,    61,
      62,    47,    77,    48,     1,    50,     4,    52,     6,     6,
      55,    56,   546,     3,    59,   523,    61,    60,     4,   527,
       6,     3,     3,     3,     1,   557,     1,   559,   566,   567,
     564,     3,     3,     6,    63,    77,    81,     6,     6,     3,
       3,     6,   104,   105,     6,     6,   108,   109,    93,    94,
       3,     6,     3,    98,    59,     3,    61,    62,    60,     6,
     590,   106,     3,   108,   109,   110,     3,    75,     3,     3,
       6,     3,   606,    33,   608,     3,     3,     3,     9,     4,
       3,     3,     3,    59,    30,    61,    62,   609,   610,     9,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,     9,     9,   108,   109,     3,    58,   641,   642,     3,
      59,    77,    88,    89,    90,    91,    92,     3,   656,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
       3,     3,   108,   109,     1,   657,     3,     3,     1,   669,
       3,     3,     9,    76,     3,     3,     9,     3,     3,    58,
      47,     3,    58,    77,     3,   687,    23,    24,    75,   681,
      23,    24,     6,     6,     9,     3,     3,     9,     9,    58,
      75,   709,     3,     6,   712,     3,     3,     3,     3,   717,
      47,   719,     3,     3,    47,     3,     3,     6,   710,     3,
       3,   713,     3,   648,   654,   681,   718,   379,   720,   663,
     505,   668,   459,    37,   738,   451,   728,   349,   740,   731,
     627,   747,   734,   457,   542,   542,   542,    32,   749,   186,
      -1,    -1,   760,    -1,    -1,   763,    -1,    -1,   766,    -1,
      -1,    -1,   770,    -1,    -1,     0,     1,   771,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    19,    20,    21,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    47,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    16,    17,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    -1,    31,    32,    -1,    34,
      35,    36,    37,    -1,    -1,    40,    -1,    -1,    43,    -1,
      -1,    -1,    -1,    48,    -1,    50,    51,    52,    53,    54,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      85,    -1,    -1,    -1,    26,    27,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    37,    -1,    -1,    -1,   104,
      -1,   106,    -1,   108,   109,   110,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    37,    -1,
      -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    37,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,
     109,   110,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    60,    61,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,   109,   110,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    37,    -1,    -1,
      -1,    -1,    -1,   106,    -1,   108,   109,   110,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,
     110,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    -1,    61,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    37,    -1,    -1,    -1,    -1,    -1,   106,
      -1,   108,   109,   110,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,   109,   110,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    37,
      -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,
      48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,
      -1,    59,    -1,    61,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    93,    94,    -1,    -1,    -1,
      98,    -1,    37,    -1,    -1,    -1,    -1,    -1,   106,    -1,
     108,   109,   110,    48,    -1,    50,    -1,    52,    -1,    -1,
      55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    93,    94,
      -1,    -1,    -1,    98,    -1,    37,    -1,    -1,    -1,    -1,
      -1,   106,    -1,   108,   109,   110,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    37,    -1,
      -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,    48,
      -1,    50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,
      59,    -1,    61,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    37,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,
     109,   110,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    37,    -1,    -1,    -1,    -1,    -1,
     106,    -1,   108,   109,   110,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    37,    -1,    -1,
      -1,    -1,    -1,   106,    -1,   108,   109,   110,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      -1,    61,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,
     110,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    -1,    61,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    37,    -1,    -1,    -1,    -1,    -1,   106,
      -1,   108,   109,   110,    48,    -1,    50,    -1,    52,     1,
      -1,    55,    56,    -1,    -1,    59,    -1,    61,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    47,    -1,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,   109,   110,    59,    -1,    61,
      62,     3,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    47,   108,   109,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    59,    -1,    61,
      62,     3,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    47,   108,   109,    -1,    -1,
      -1,    -1,     3,    -1,    -1,    -1,    -1,    59,    -1,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,   108,   109,    59,    -1,
      61,    62,     3,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    77,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    47,   108,   109,    -1,
      -1,    -1,    -1,     3,    -1,    -1,    -1,    -1,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,     3,    -1,   108,   109,    59,
      -1,    61,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,     3,    -1,   108,   109,
      -1,    59,    -1,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,     3,    -1,
     108,   109,    59,    -1,    61,    62,    -1,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,     3,
      -1,   108,   109,    -1,    59,    -1,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,     3,    -1,   108,   109,    59,    -1,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,     3,    -1,   108,   109,    -1,    59,    -1,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,     3,    -1,   108,   109,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,     3,    -1,   108,   109,    -1,
      59,    -1,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,     3,    -1,   108,
     109,    59,    -1,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    -1,    59,    -1,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    47,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      60,    61,    -1,    63,    -1,    -1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    -1,    93,    94,    -1,    -1,    59,    98,    61,
      62,    37,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,
     110,    47,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    81,   108,   109,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,
      -1,    -1,    98,    -1,    37,    -1,   102,    -1,    -1,    -1,
     106,    -1,   108,   109,   110,    48,    -1,    50,    -1,    52,
      -1,    -1,    55,    56,    -1,    -1,    59,    60,    61,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    37,    -1,    -1,
      -1,    -1,    -1,   106,    -1,   108,   109,   110,    48,    -1,
      50,    -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,
      60,    61,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,
      37,    -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,
     110,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    58,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    37,    -1,    -1,    -1,    -1,    -1,   106,
      -1,   108,   109,   110,    48,    -1,    50,    -1,    52,    -1,
      -1,    55,    56,    -1,    -1,    59,    60,    61,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    37,    -1,    -1,    -1,
      -1,    -1,   106,    -1,   108,   109,   110,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    47,    -1,
      -1,    -1,    -1,    -1,    -1,   106,    -1,   108,   109,   110,
      59,    -1,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    47,    -1,   108,
     109,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    59,
      60,    61,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    47,    -1,   108,   109,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    59,    60,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    58,
      59,    -1,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,   108,
     109,    59,    60,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
     108,   109,    59,    -1,    61,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,   108,   109,    59,    60,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,   108,   109,    59,    60,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,   108,   109,    59,    -1,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,   108,   109,    59,    -1,    61,    62,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    75,    -1,    77,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,   108,   109,    59,    -1,    61,
      62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,   108,   109,    59,    -1,
      61,    62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,   108,   109,    59,
      -1,    61,    62,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,   108,   109,
      59,    -1,    61,    62,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,   108,
     109
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   112,   113,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    57,    59,    61,
      81,    85,    93,    94,    98,   104,   106,   108,   109,   110,
     114,   115,   117,   118,   119,   121,   122,   124,   125,   126,
     128,   129,   137,   138,   139,   144,   145,   146,   153,   155,
     168,   170,   179,   180,   182,   189,   190,   191,   193,   195,
     203,   204,   205,   206,   208,   210,   213,   214,   215,   218,
     236,   241,   246,   250,   251,   252,   253,   254,   255,   256,
     258,   261,   263,   266,   267,   268,   269,   270,     3,     3,
       1,   120,   252,     1,    48,   254,     1,     3,     1,     3,
      14,     1,   254,     1,   252,   272,     1,   254,     1,     1,
     254,     1,   254,   270,     1,     3,    47,     1,   254,     1,
       6,     1,     6,     1,     3,   254,   247,   264,     1,     6,
       7,     1,   254,   256,     1,     6,     1,     3,    47,     1,
     254,     1,     3,     6,   207,     1,     6,   207,   209,     1,
       6,   211,   212,     1,     6,   259,   262,     1,   254,    60,
     254,   271,    47,     6,   254,    47,    60,    63,   254,   270,
     273,   254,     1,     3,   270,   254,   254,   254,     1,     3,
     270,   254,   254,   254,   110,   252,   123,    32,    34,    48,
     118,   127,   118,   154,   169,   181,    49,   198,   201,   202,
     118,     1,    59,     1,     6,   216,   217,   216,     3,    59,
      61,    62,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    78,    79,    80,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   108,
     109,   255,    75,    77,     3,     3,    77,    75,     3,    47,
       3,    47,     3,     3,     3,     3,    47,     3,    47,     3,
      77,    90,     3,     3,     3,     3,     3,     3,     1,    76,
       3,   118,     3,     3,     3,   219,     3,   242,     3,     3,
       6,   248,   249,     1,     6,   196,   197,   265,     3,     3,
       3,     3,     3,     3,    75,     3,    47,     3,     3,    90,
       3,     3,    77,     3,    77,     3,     3,    75,     3,    77,
       3,     1,    59,   260,   260,     3,     3,     1,    60,   254,
     237,    58,    60,   254,    60,    47,    63,     1,    60,     1,
      60,    77,     3,     3,     3,     3,   136,   136,   156,   171,
     136,     1,     3,    47,   136,   199,   200,     3,     1,   196,
       3,     3,    77,     9,   216,     3,    58,   270,   102,   254,
       6,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,     1,   254,   254,   254,   254,   254,   254,
     254,   254,   254,     6,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   252,   254,   252,     1,   254,     3,   270,     1,    59,
     220,   221,     1,    33,   223,   243,     3,    77,     3,    77,
      63,     1,   251,     1,   254,     6,     6,     4,     6,    98,
     116,   212,     3,   196,   198,   198,     3,    60,    60,   254,
     254,    60,   254,     9,   118,    16,    17,   130,   132,   133,
     135,     1,     3,    23,    24,   157,   162,   164,     1,     3,
      23,   162,   172,    30,   183,   184,   185,   186,     3,    47,
       9,   136,   118,   194,     1,    58,     6,     3,     3,     1,
      58,   254,    60,    77,     1,    47,     3,    77,    75,     3,
       3,    47,     3,     3,   196,   229,   223,     3,     6,   224,
     225,     3,   244,     1,   249,   197,   254,     3,     3,     3,
       3,     4,     1,    58,   136,   136,   238,    47,    60,    63,
       3,     1,     3,     1,   254,     9,   131,   134,     3,     1,
       6,     7,     8,   116,   166,   167,     1,     9,   163,     3,
       1,     4,     6,   177,   178,     9,     1,     3,     4,     6,
      90,   187,   188,     9,   185,   136,     3,     9,    58,   192,
       3,    47,   257,    60,   270,     1,   254,   270,   254,   140,
     141,     1,    58,     3,     6,    38,    48,    49,    92,   190,
     230,   231,   233,   234,     3,    59,   226,    77,     3,   190,
     231,   233,   234,   245,     3,     9,     9,     3,     6,     9,
     239,   254,   254,     3,     3,     3,     3,   136,   136,     3,
      47,    76,     3,    47,    77,     3,     3,    47,   165,     3,
      47,     3,    47,    77,     3,     3,   252,     3,    77,    90,
       3,     3,    47,    58,    58,     3,    19,    20,    21,   118,
     142,   143,   147,   149,   151,   118,   222,    75,     3,     6,
       1,     6,    81,   235,     9,     6,    27,   227,   228,   251,
     225,     9,     3,    75,    77,   240,     3,    60,   130,   160,
     161,   116,   158,   159,   167,   136,   118,   175,   176,   173,
     174,   178,     3,   188,   252,     3,     1,     3,    47,     1,
       3,    47,     1,     3,    47,     9,   142,    58,   254,   232,
      75,     3,     6,     3,    77,     3,    58,    77,     3,   251,
     136,   118,   136,   118,   136,   118,   136,   118,     3,     3,
     148,   118,     3,   150,   118,     3,   152,   118,     3,     3,
     198,   254,     6,    81,   228,   240,   136,   136,   136,   136,
       3,     6,     9,     9,     9,     9,     3,     3,     3,     3
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
        case 6:
#line 211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); ;}
    break;

  case 7:
#line 212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); ;}
    break;

  case 8:
#line 217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      ;}
    break;

  case 10:
#line 223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 11:
#line 228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 12:
#line 233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 13:
#line 238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      ;}
    break;

  case 19:
#line 250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); ;}
    break;

  case 20:
#line 255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 21:
#line 261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      ;}
    break;

  case 22:
#line 267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      ;}
    break;

  case 23:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); ;}
    break;

  case 24:
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; ;}
    break;

  case 25:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; ;}
    break;

  case 26:
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; ;}
    break;

  case 27:
#line 278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; ;}
    break;

  case 28:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;;}
    break;

  case 29:
#line 284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); ;}
    break;

  case 30:
#line 286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      ;}
    break;

  case 31:
#line 292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (6)].fal_adecl)->size() != (yyvsp[(5) - (6)].fal_adecl)->size() + 1 )
         {
            COMPILER->raiseError(Falcon::e_unpack_size );
         }
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (6)].fal_adecl) );

         COMPILER->defineVal( first );
         (yyvsp[(5) - (6)].fal_adecl)->pushFront( (yyvsp[(3) - (6)].fal_val) );
         Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (6)].fal_adecl) );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, second ) ) );
      ;}
    break;

  case 51:
#line 328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   ;}
    break;

  case 52:
#line 334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   ;}
    break;

  case 53:
#line 343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; ;}
    break;

  case 54:
#line 345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); ;}
    break;

  case 55:
#line 349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      ;}
    break;

  case 56:
#line 356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      ;}
    break;

  case 57:
#line 363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      ;}
    break;

  case 58:
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 59:
#line 372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 60:
#line 373 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; ;}
    break;

  case 61:
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 62:
#line 378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; ;}
    break;

  case 63:
#line 379 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 64:
#line 383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      ;}
    break;

  case 65:
#line 391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 66:
#line 398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 67:
#line 408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 68:
#line 409 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; ;}
    break;

  case 69:
#line 413 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 70:
#line 414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; ;}
    break;

  case 73:
#line 421 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      ;}
    break;

  case 76:
#line 431 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); ;}
    break;

  case 77:
#line 436 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      ;}
    break;

  case 79:
#line 448 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 80:
#line 449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; ;}
    break;

  case 82:
#line 454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   ;}
    break;

  case 83:
#line 461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      ;}
    break;

  case 84:
#line 470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 85:
#line 478 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      ;}
    break;

  case 86:
#line 488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      ;}
    break;

  case 87:
#line 497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 88:
#line 506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 89:
#line 522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 90:
#line 530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
          Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (5)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( CURRENT_LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (5)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( CURRENT_LINE, new Falcon::Value(decl), (yyvsp[(4) - (5)].fal_val) );

         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      ;}
    break;

  case 91:
#line 546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      ;}
    break;

  case 92:
#line 556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 93:
#line 561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 96:
#line 573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      ;}
    break;

  case 100:
#line 587 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getLoop());
         if ( f == 0 || f->type() != Falcon::Statement::t_forin )
         {
            COMPILER->raiseError( Falcon::e_syn_fordot );
            delete (yyvsp[(2) - (3)].fal_val);
            (yyval.fal_stat) = 0;
         }
         else {
            (yyval.fal_stat) = new Falcon::StmtFordot( LINE, (yyvsp[(2) - (3)].fal_val) );
         }
      ;}
    break;

  case 101:
#line 600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 102:
#line 608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 103:
#line 612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 104:
#line 618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 105:
#line 624 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      ;}
    break;

  case 106:
#line 631 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 107:
#line 636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 108:
#line 645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   ;}
    break;

  case 109:
#line 654 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      ;}
    break;

  case 110:
#line 666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 111:
#line 668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 112:
#line 677 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); ;}
    break;

  case 113:
#line 681 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      ;}
    break;

  case 114:
#line 693 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 115:
#line 694 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 116:
#line 703 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); ;}
    break;

  case 117:
#line 707 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 // Apparently you get a segfault without it.
		 // (Note that the formiddle: version below does *not* need it
		 f->middleBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->middleBlock() );
      ;}
    break;

  case 118:
#line 721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); ;}
    break;

  case 119:
#line 723 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      ;}
    break;

  case 120:
#line 732 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); ;}
    break;

  case 121:
#line 736 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 122:
#line 744 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 123:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 124:
#line 755 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 127:
#line 764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); ;}
    break;

  case 129:
#line 770 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 131:
#line 780 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 132:
#line 788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 133:
#line 792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 135:
#line 804 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 136:
#line 814 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 138:
#line 823 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      ;}
    break;

  case 142:
#line 837 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); ;}
    break;

  case 144:
#line 841 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      ;}
    break;

  case 147:
#line 853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      ;}
    break;

  case 148:
#line 862 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 149:
#line 874 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 150:
#line 885 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 151:
#line 896 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 152:
#line 916 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      ;}
    break;

  case 153:
#line 924 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      ;}
    break;

  case 154:
#line 933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); ;}
    break;

  case 155:
#line 935 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      ;}
    break;

  case 158:
#line 944 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); ;}
    break;

  case 160:
#line 950 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 162:
#line 960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 163:
#line 969 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 164:
#line 973 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 166:
#line 985 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      ;}
    break;

  case 167:
#line 995 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      ;}
    break;

  case 171:
#line 1009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 172:
#line 1021 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 )
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 173:
#line 1042 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_adecl), (yyvsp[(2) - (5)].fal_adecl) );
      ;}
    break;

  case 174:
#line 1046 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      ;}
    break;

  case 175:
#line 1050 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; ;}
    break;

  case 176:
#line 1058 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   ;}
    break;

  case 177:
#line 1065 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      ;}
    break;

  case 178:
#line 1075 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      ;}
    break;

  case 180:
#line 1084 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); ;}
    break;

  case 186:
#line 1104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }
         // but continue by pushing this new context
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 187:
#line 1122 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );

         // if we have already a default, raise an error
         if( t->defaultHandler() != 0 )
         {
            COMPILER->raiseError(Falcon::e_catch_adef );
         }

         // but continue by pushing this new context
         COMPILER->defineVal( (yyvsp[(3) - (4)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(3) - (4)].fal_val) );
         t->defaultHandler( lst ); // will delete the previous one

         COMPILER->pushContextSet( &lst->children() );
      ;}
    break;

  case 188:
#line 1142 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 189:
#line 1152 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      ;}
    break;

  case 190:
#line 1163 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   ;}
    break;

  case 193:
#line 1176 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 194:
#line 1188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         Falcon::Value *val = new Falcon::Value( sym );

         if ( ! stmt->addSymbolCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      ;}
    break;

  case 195:
#line 1210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 196:
#line 1211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; ;}
    break;

  case 197:
#line 1223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 198:
#line 1229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      ;}
    break;

  case 200:
#line 1238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 201:
#line 1239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 202:
#line 1242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); ;}
    break;

  case 204:
#line 1247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 205:
#line 1248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 206:
#line 1255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // if we are in a class, I have to create the symbol classname.functionname
         Falcon::Statement *parent = COMPILER->getContext();
         Falcon::String *func_name;
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::String complete_name = stmt_cls->symbol()->name() + "." + *(yyvsp[(2) - (2)].stringp);
            func_name = COMPILER->addString( complete_name );
         }
         else
            func_name = (yyvsp[(2) - (2)].stringp);

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( func_name );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( func_name );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         // and eventually add it as a class property
         if ( parent != 0 && parent->type() == Falcon::Statement::t_class ) {
            Falcon::StmtClass *stmt_cls = static_cast< Falcon::StmtClass *>( parent );
            Falcon::ClassDef *cd = stmt_cls->symbol()->getClassDef();
            if ( cd->hasProperty( *(yyvsp[(2) - (2)].stringp) ) ) {
               COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (2)].stringp) );
            }
            else {
                cd->addProperty( (yyvsp[(2) - (2)].stringp), new Falcon::VarDef( sym ) );
            }
         }

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 210:
#line 1316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym != 0 ) {
            COMPILER->raiseError(Falcon::e_already_def, sym->name() );
         }
         else {
            Falcon::FuncDef *func = COMPILER->getFunction();
            Falcon::Symbol *sym = new Falcon::Symbol( COMPILER->module(), (yyvsp[(1) - (1)].stringp) );
            COMPILER->module()->addSymbol( sym );
            func->addParameter( sym );
         }
      ;}
    break;

  case 212:
#line 1333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 213:
#line 1339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 214:
#line 1344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      ;}
    break;

  case 215:
#line 1350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      ;}
    break;

  case 217:
#line 1359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); ;}
    break;

  case 219:
#line 1364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); ;}
    break;

  case 220:
#line 1374 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 221:
#line 1377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; ;}
    break;

  case 222:
#line 1386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      ;}
    break;

  case 223:
#line 1393 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // define the expression anyhow so we don't have fake errors below
         if ( COMPILER->getFunction() == 0 )
         {
            COMPILER->raiseError(Falcon::e_pass_outside );
            /*delete $2;
            delete $4;*/
            (yyval.fal_stat) = 0;
         }
         else {
            COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         }
      ;}
    break;

  case 224:
#line 1408 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 225:
#line 1414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 226:
#line 1426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 227:
#line 1436 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 228:
#line 1441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 229:
#line 1453 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 230:
#line 1462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 231:
#line 1469 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 232:
#line 1477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 233:
#line 1482 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      ;}
    break;

  case 234:
#line 1490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 235:
#line 1494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 236:
#line 1502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 237:
#line 1507 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->imported(true);
      ;}
    break;

  case 238:
#line 1519 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      ;}
    break;

  case 239:
#line 1524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     ;}
    break;

  case 242:
#line 1537 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      ;}
    break;

  case 243:
#line 1541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      ;}
    break;

  case 244:
#line 1555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 245:
#line 1562 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      ;}
    break;

  case 247:
#line 1570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); ;}
    break;

  case 249:
#line 1574 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); ;}
    break;

  case 251:
#line 1580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         ;}
    break;

  case 252:
#line 1584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         ;}
    break;

  case 255:
#line 1593 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   ;}
    break;

  case 256:
#line 1604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );
      ;}
    break;

  case 257:
#line 1638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      ;}
    break;

  case 259:
#line 1666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      ;}
    break;

  case 262:
#line 1674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 263:
#line 1675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      ;}
    break;

  case 268:
#line 1692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            if ( (yyvsp[(2) - (2)].fal_adecl) != 0 )
            {
               // save the carried
               Falcon::ListElement *iter = (yyvsp[(2) - (2)].fal_adecl)->begin();
               while( iter != 0 )
               {
                  Falcon::Value *val = (Falcon::Value *) iter->data();
                  idef->addParameter( val->genVarDefSym() );
                  iter = iter->next();
               }

               // dispose of the carrier
               delete (yyvsp[(2) - (2)].fal_adecl);
            }
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
         }
      ;}
    break;

  case 269:
#line 1725 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; ;}
    break;

  case 270:
#line 1730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   ;}
    break;

  case 271:
#line 1736 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 272:
#line 1737 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 274:
#line 1743 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      ;}
    break;

  case 275:
#line 1751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 279:
#line 1761 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 280:
#line 1764 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 282:
#line 1786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         if( cls->initGiven() ) {
            COMPILER->raiseError(Falcon::e_init_given );
         }
         else
         {
            cls->initGiven( true );
            Falcon::StmtFunction *func = cls->ctorFunction();
            if ( func == 0 ) {
               func = COMPILER->buildCtorFor( cls );
            }

            // prepare the statement allocation context
            COMPILER->pushContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      ;}
    break;

  case 283:
#line 1810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 284:
#line 1821 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(4) - (5)].fal_val)->genVarDef();

      if ( def != 0 ) {
         Falcon::String prop_name = cls->symbol()->name() + "." + *(yyvsp[(2) - (5)].stringp);
         Falcon::Symbol *sym = COMPILER->addGlobalVar( COMPILER->addString(prop_name), def );
         if( clsdef->hasProperty( *(yyvsp[(2) - (5)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(2) - (5)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(2) - (5)].stringp), new Falcon::VarDef( Falcon::VarDef::t_reference, sym) );
      }
      else {
         COMPILER->raiseError(Falcon::e_static_const );
      }
      delete (yyvsp[(4) - (5)].fal_val); // the expression is not needed anymore
      (yyval.fal_stat) = 0; // we don't add any statement
   ;}
    break;

  case 285:
#line 1843 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->checkLocalUndefined();
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
      Falcon::VarDef *def = (yyvsp[(3) - (4)].fal_val)->genVarDef();

      if ( def != 0 ) {
         if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), def );
         delete (yyvsp[(3) - (4)].fal_val); // the expression is not needed anymore
         (yyval.fal_stat) = 0; // we don't add any statement
      }
      else {
         // create anyhow a nil property
          if( clsdef->hasProperty( *(yyvsp[(1) - (4)].stringp) ) )
            COMPILER->raiseError(Falcon::e_prop_adef, *(yyvsp[(1) - (4)].stringp) );
         else
            clsdef->addProperty( (yyvsp[(1) - (4)].stringp), new Falcon::VarDef() );
         // but also prepare a statement to be executed by the auto-constructor.
         (yyval.fal_stat) = new Falcon::StmtVarDef( LINE, (yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
   ;}
    break;

  case 288:
#line 1873 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   ;}
    break;

  case 289:
#line 1880 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      ;}
    break;

  case 290:
#line 1888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      ;}
    break;

  case 291:
#line 1894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      ;}
    break;

  case 292:
#line 1900 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      ;}
    break;

  case 293:
#line 1913 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
            sym->setEnum( true );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setClass( def );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), sym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         // We don't have a context set here
         COMPILER->pushFunction( def );

         COMPILER->resetEnum();
      ;}
    break;

  case 294:
#line 1947 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      ;}
    break;

  case 298:
#line 1964 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      ;}
    break;

  case 299:
#line 1969 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      ;}
    break;

  case 302:
#line 1984 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef( 0, 0 );
         // the SYMBOL which names the function goes in the old symbol table, while the parameters
         // will go in the new symbol table.

         // we create a special symbol for the class.
         Falcon::String cl_name = "%";
         cl_name += *(yyvsp[(2) - (2)].stringp);
         Falcon::Symbol *clsym = COMPILER->addGlobalSymbol( COMPILER->addString( cl_name ) );
         clsym->setClass( def );

         // find the global symbol for this.
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( (yyvsp[(2) - (2)].stringp) );

         // Not defined?
         if( sym == 0 ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) );
         }
         else if ( sym->isFunction() || sym->isClass() ) {
            COMPILER->raiseError(Falcon::e_already_def,  sym->name() );
         }

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setInstance( clsym );

         Falcon::StmtClass *cls = new Falcon::StmtClass( COMPILER->lexer()->line(), clsym );
         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 303:
#line 2024 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               Falcon::StmtFunction *func = func = COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      ;}
    break;

  case 305:
#line 2049 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      ;}
    break;

  case 309:
#line 2061 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   ;}
    break;

  case 310:
#line 2064 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
      if ( cls->initGiven() ) {
         COMPILER->raiseError(Falcon::e_prop_pinit );
      }
      COMPILER->checkLocalUndefined();
      // have we got a complex property statement?
      if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
      {
         // as we didn't push the class context set, we have to do it by ourselves
         // see if the class has already a constructor.
         Falcon::StmtFunction *ctor_stmt = cls->ctorFunction();
         if ( ctor_stmt == 0 ) {
            ctor_stmt = COMPILER->buildCtorFor( cls );
         }

         ctor_stmt->statements().push_back( (yyvsp[(1) - (1)].fal_stat) );  // this goes directly in the auto constructor.
      }
   ;}
    break;

  case 312:
#line 2092 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      ;}
    break;

  case 313:
#line 2097 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      ;}
    break;

  case 316:
#line 2112 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      ;}
    break;

  case 317:
#line 2119 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      ;}
    break;

  case 318:
#line 2134 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); ;}
    break;

  case 319:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); ;}
    break;

  case 320:
#line 2136 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; ;}
    break;

  case 321:
#line 2146 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); ;}
    break;

  case 322:
#line 2147 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); ;}
    break;

  case 323:
#line 2148 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); ;}
    break;

  case 324:
#line 2149 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); ;}
    break;

  case 325:
#line 2150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); ;}
    break;

  case 326:
#line 2151 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); ;}
    break;

  case 327:
#line 2156 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *val;
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if( sym == 0 ) {
            val = new Falcon::Value();
            val->setSymdef( (yyvsp[(1) - (1)].stringp) );
            // warning: the symbol is still undefined.
            COMPILER->addSymdef( val );
         }
         else {
            val = new Falcon::Value( sym );
         }
         (yyval.fal_val) = val;
     ;}
    break;

  case 329:
#line 2174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); ;}
    break;

  case 330:
#line 2175 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); ;}
    break;

  case 333:
#line 2188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 334:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 335:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 336:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 337:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 338:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 339:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 340:
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 341:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 342:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 343:
#line 2198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 344:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 345:
#line 2200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 346:
#line 2201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 347:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 348:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 349:
#line 2204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); ;}
    break;

  case 350:
#line 2205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 351:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 352:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 353:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 354:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 355:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 356:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 357:
#line 2212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 358:
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 359:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 360:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 361:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 362:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 363:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); ;}
    break;

  case 364:
#line 2219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); ;}
    break;

  case 365:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); ;}
    break;

  case 366:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 367:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); ;}
    break;

  case 374:
#line 2230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 375:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   ;}
    break;

  case 376:
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   ;}
    break;

  case 377:
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 378:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      ;}
    break;

  case 381:
#line 2262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   ;}
    break;

  case 382:
#line 2267 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   ;}
    break;

  case 383:
#line 2274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 384:
#line 2275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 385:
#line 2276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 386:
#line 2277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 387:
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 388:
#line 2279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 389:
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 390:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 391:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 392:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 393:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); ;}
    break;

  case 394:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);;}
    break;

  case 395:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      ;}
    break;

  case 396:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      ;}
    break;

  case 397:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      ;}
    break;

  case 398:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      ;}
    break;

  case 399:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      ;}
    break;

  case 400:
#line 2309 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      ;}
    break;

  case 401:
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      ;}
    break;

  case 402:
#line 2319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); ;}
    break;

  case 403:
#line 2320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 404:
#line 2329 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 405:
#line 2363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         ;}
    break;

  case 407:
#line 2371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      ;}
    break;

  case 408:
#line 2375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      ;}
    break;

  case 409:
#line 2382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 410:
#line 2415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         ;}
    break;

  case 411:
#line 2427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0 );
         // set the def as a lambda.
         COMPILER->incLambdaCount();
         COMPILER->incClosureContext();
         int id = COMPILER->lambdaCount();
         // find the global symbol for this.
         char buf[48];
         sprintf( buf, "_lambda#_id_%d", id );
         Falcon::String *name = COMPILER->addString( buf );
         Falcon::Symbol *sym = COMPILER->searchGlobalSymbol( name );

         // Not defined?
         fassert( sym == 0 );
         sym = COMPILER->addGlobalSymbol( name );

         // anyhow, also in case of error, destroys the previous information to allow a correct parsing
         // of the rest.
         sym->setFunction( def );

         Falcon::StmtFunction *func = new Falcon::StmtFunction( COMPILER->lexer()->line(), sym );
         COMPILER->addFunction( func );
         func->setLambda( id );
         // prepare the statement allocation context
         COMPILER->pushContext( func );
         COMPILER->pushFunctionContext( func );
         COMPILER->pushContextSet( &func->statements() );
         COMPILER->pushFunction( def );
      ;}
    break;

  case 412:
#line 2459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         ;}
    break;

  case 414:
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      ;}
    break;

  case 415:
#line 2480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   ;}
    break;

  case 416:
#line 2485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 417:
#line 2492 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   ;}
    break;

  case 418:
#line 2499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      ;}
    break;

  case 419:
#line 2508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); ;}
    break;

  case 420:
#line 2510 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      ;}
    break;

  case 421:
#line 2514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      ;}
    break;

  case 422:
#line 2521 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); ;}
    break;

  case 423:
#line 2523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 424:
#line 2527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      ;}
    break;

  case 425:
#line 2535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); ;}
    break;

  case 426:
#line 2536 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); ;}
    break;

  case 427:
#line 2538 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      ;}
    break;

  case 428:
#line 2545 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 429:
#line 2546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); ;}
    break;

  case 430:
#line 2550 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); ;}
    break;

  case 431:
#line 2551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); ;}
    break;

  case 432:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      ;}
    break;

  case 433:
#line 2561 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      ;}
    break;

  case 434:
#line 2568 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); ;}
    break;

  case 435:
#line 2569 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); ;}
    break;


/* Line 1267 of yacc.c.  */
#line 6247 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2573 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */



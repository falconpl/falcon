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
     CAP_CAP = 350,
     VBAR_VBAR = 351,
     AMPER_AMPER = 352,
     MINUS = 353,
     PLUS = 354,
     PERCENT = 355,
     SLASH = 356,
     STAR = 357,
     POW = 358,
     SHR = 359,
     SHL = 360,
     TILDE = 361,
     NEG = 362,
     AMPER = 363,
     BANG = 364,
     DECREMENT = 365,
     INCREMENT = 366,
     DOLLAR = 367
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
#define CAP_CAP 350
#define VBAR_VBAR 351
#define AMPER_AMPER 352
#define MINUS 353
#define PLUS 354
#define PERCENT 355
#define SLASH 356
#define STAR 357
#define POW 358
#define SHR 359
#define SHL 360
#define TILDE 361
#define NEG 362
#define AMPER 363
#define BANG 364
#define DECREMENT 365
#define INCREMENT 366
#define DOLLAR 367




/* Copy the first part of user declarations.  */
#line 17 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"


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
#line 61 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
/* Line 187 of yacc.c.  */
#line 385 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 398 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6142

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  113
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  164
/* YYNRULES -- Number of rules.  */
#define YYNRULES  443
/* YYNRULES -- Number of states.  */
#define YYNSTATES  806

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   367

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
     105,   106,   107,   108,   109,   110,   111,   112
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
     744,   748,   752,   754,   758,   762,   768,   776,   781,   786,
     793,   797,   799,   801,   803,   807,   811,   815,   817,   821,
     825,   829,   833,   838,   842,   845,   849,   852,   856,   857,
     859,   863,   866,   870,   873,   874,   883,   887,   890,   891,
     895,   896,   902,   903,   906,   908,   912,   915,   916,   920,
     922,   926,   928,   930,   932,   933,   936,   938,   940,   942,
     944,   945,   953,   959,   964,   965,   969,   973,   975,   978,
     982,   987,   988,   996,   997,  1000,  1002,  1007,  1010,  1012,
    1014,  1015,  1024,  1027,  1030,  1031,  1034,  1036,  1038,  1040,
    1042,  1043,  1048,  1050,  1054,  1058,  1060,  1063,  1067,  1071,
    1073,  1075,  1077,  1079,  1081,  1083,  1085,  1087,  1089,  1091,
    1093,  1095,  1098,  1102,  1106,  1110,  1114,  1118,  1122,  1126,
    1130,  1134,  1138,  1142,  1145,  1149,  1152,  1155,  1158,  1161,
    1165,  1169,  1173,  1177,  1181,  1185,  1189,  1192,  1196,  1200,
    1204,  1208,  1212,  1215,  1218,  1221,  1224,  1226,  1228,  1230,
    1232,  1234,  1236,  1239,  1241,  1246,  1252,  1256,  1258,  1260,
    1264,  1270,  1274,  1278,  1282,  1286,  1290,  1294,  1298,  1302,
    1306,  1310,  1314,  1318,  1322,  1327,  1332,  1338,  1346,  1351,
    1355,  1356,  1363,  1364,  1371,  1376,  1380,  1383,  1384,  1391,
    1392,  1398,  1400,  1403,  1409,  1415,  1420,  1424,  1427,  1431,
    1435,  1438,  1442,  1446,  1450,  1454,  1459,  1461,  1465,  1467,
    1470,  1472,  1476,  1480
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     114,     0,    -1,   115,    -1,    -1,   115,   116,    -1,   117,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   119,    -1,
     213,    -1,   192,    -1,   221,    -1,   244,    -1,   120,    -1,
     207,    -1,   208,    -1,   210,    -1,   216,    -1,     4,    -1,
      98,     4,    -1,    39,     6,     3,    -1,    39,     7,     3,
      -1,    39,     1,     3,    -1,   121,    -1,     3,    -1,    48,
       1,     3,    -1,    34,     1,     3,    -1,    32,     1,     3,
      -1,     1,     3,    -1,   257,     3,    -1,   273,    75,   257,
       3,    -1,   273,    75,   257,    77,   273,     3,    -1,   123,
      -1,   124,    -1,   141,    -1,   155,    -1,   170,    -1,   128,
      -1,   139,    -1,   140,    -1,   181,    -1,   182,    -1,   191,
      -1,   253,    -1,   249,    -1,   205,    -1,   206,    -1,   146,
      -1,   147,    -1,   148,    -1,   239,    -1,   255,    75,   257,
      -1,   122,    77,   255,    75,   257,    -1,    10,   122,     3,
      -1,    10,     1,     3,    -1,    -1,   126,   125,   138,     9,
       3,    -1,   127,   120,    -1,    11,   257,     3,    -1,    53,
      -1,    11,     1,     3,    -1,    11,   257,    47,    -1,    53,
      47,    -1,    11,     1,    47,    -1,    -1,   130,   129,   138,
     132,     9,     3,    -1,   131,   120,    -1,    15,   257,     3,
      -1,    15,     1,     3,    -1,    15,   257,    47,    -1,    15,
       1,    47,    -1,    -1,   135,    -1,    -1,   134,   133,   138,
      -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,   137,
     136,   138,   132,    -1,    17,   257,     3,    -1,    17,     1,
       3,    -1,    -1,   138,   120,    -1,    12,     3,    -1,    12,
       1,     3,    -1,    13,     3,    -1,    13,    14,     3,    -1,
      13,     1,     3,    -1,    -1,    18,   275,    90,   257,     3,
     142,   144,     9,     3,    -1,    -1,    18,   275,    90,   257,
      47,   143,   120,    -1,    18,   275,    90,     1,     3,    -1,
      18,     1,     3,    -1,    -1,   145,   144,    -1,   120,    -1,
     149,    -1,   151,    -1,   153,    -1,    51,   257,     3,    -1,
      51,     1,     3,    -1,   104,   273,     3,    -1,   104,     3,
      -1,    85,   273,     3,    -1,    85,     3,    -1,   104,     1,
       3,    -1,    85,     1,     3,    -1,    57,    -1,    -1,    19,
       3,   150,   138,     9,     3,    -1,    19,    47,   120,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   152,   138,     9,
       3,    -1,    20,    47,   120,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   154,   138,     9,     3,    -1,    21,    47,
     120,    -1,    21,     1,     3,    -1,    -1,   157,   156,   158,
     164,     9,     3,    -1,    22,   257,     3,    -1,    22,     1,
       3,    -1,    -1,   158,   159,    -1,   158,     1,     3,    -1,
       3,    -1,    -1,    23,   168,     3,   160,   138,    -1,    -1,
      23,   168,    47,   161,   120,    -1,    -1,    23,     1,     3,
     162,   138,    -1,    -1,    23,     1,    47,   163,   120,    -1,
      -1,    -1,   166,   165,   167,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   138,    -1,    47,   120,    -1,   169,    -1,
     168,    77,   169,    -1,     8,    -1,   118,    -1,     7,    -1,
     118,    76,   118,    -1,     6,    -1,    -1,   172,   171,   173,
     164,     9,     3,    -1,    25,   257,     3,    -1,    25,     1,
       3,    -1,    -1,   173,   174,    -1,   173,     1,     3,    -1,
       3,    -1,    -1,    23,   179,     3,   175,   138,    -1,    -1,
      23,   179,    47,   176,   120,    -1,    -1,    23,     1,     3,
     177,   138,    -1,    -1,    23,     1,    47,   178,   120,    -1,
     180,    -1,   179,    77,   180,    -1,    -1,     4,    -1,     6,
      -1,    28,   273,    76,   273,     3,    -1,    28,   273,     1,
       3,    -1,    28,     1,     3,    -1,    29,    47,   120,    -1,
      -1,   184,   183,   138,   185,     9,     3,    -1,    29,     3,
      -1,    29,     1,     3,    -1,    -1,   186,    -1,   187,    -1,
     186,   187,    -1,   188,   138,    -1,    30,     3,    -1,    30,
      90,   255,     3,    -1,    30,   189,     3,    -1,    30,   189,
      90,   255,     3,    -1,    30,     1,     3,    -1,   190,    -1,
     189,    77,   190,    -1,     4,    -1,     6,    -1,    31,   257,
       3,    -1,    31,     1,     3,    -1,   193,   200,   138,     9,
       3,    -1,   195,   120,    -1,   197,    59,   198,    58,     3,
      -1,    -1,   197,    59,   198,     1,   194,    58,     3,    -1,
     197,     1,     3,    -1,   197,    59,   198,    58,    47,    -1,
      -1,   197,    59,     1,   196,    58,    47,    -1,    48,     6,
      -1,    -1,   199,    -1,   198,    77,   199,    -1,     6,    -1,
      -1,    -1,   203,   201,   138,     9,     3,    -1,    -1,   204,
     202,   120,    -1,    49,     3,    -1,    49,     1,     3,    -1,
      49,    47,    -1,    49,     1,    47,    -1,    40,   259,     3,
      -1,    40,     1,     3,    -1,    43,   257,     3,    -1,    43,
     257,    90,   257,     3,    -1,    43,   257,    90,     1,     3,
      -1,    43,     1,     3,    -1,    41,     6,    75,   254,     3,
      -1,    41,     6,    75,     1,     3,    -1,    41,     1,     3,
      -1,    44,     3,    -1,    44,   209,     3,    -1,    44,     1,
       3,    -1,     6,    -1,   209,    77,     6,    -1,    45,   212,
       3,    -1,    45,   212,    33,   211,     3,    -1,    45,   212,
      33,   211,    75,     6,     3,    -1,    45,   212,     1,     3,
      -1,    45,    33,   211,     3,    -1,    45,    33,   211,    75,
       6,     3,    -1,    45,     1,     3,    -1,     6,    -1,     7,
      -1,     6,    -1,   212,    77,     6,    -1,    46,   214,     3,
      -1,    46,     1,     3,    -1,   215,    -1,   214,    77,   215,
      -1,     6,    75,     6,    -1,     6,    75,     7,    -1,     6,
      75,   118,    -1,   217,   220,     9,     3,    -1,   218,   219,
       3,    -1,    42,     3,    -1,    42,     1,     3,    -1,    42,
      47,    -1,    42,     1,    47,    -1,    -1,     6,    -1,   219,
      77,     6,    -1,   219,     3,    -1,   220,   219,     3,    -1,
       1,     3,    -1,    -1,    32,     6,   222,   223,   232,   237,
       9,     3,    -1,   224,   226,     3,    -1,     1,     3,    -1,
      -1,    59,   198,    58,    -1,    -1,    59,   198,     1,   225,
      58,    -1,    -1,    33,   227,    -1,   228,    -1,   227,    77,
     228,    -1,     6,   229,    -1,    -1,    59,   230,    58,    -1,
     231,    -1,   230,    77,   231,    -1,   254,    -1,     6,    -1,
      27,    -1,    -1,   232,   233,    -1,     3,    -1,   192,    -1,
     236,    -1,   234,    -1,    -1,    38,     3,   235,   200,   138,
       9,     3,    -1,    49,     6,    75,   257,     3,    -1,     6,
      75,   257,     3,    -1,    -1,    92,   238,     3,    -1,    92,
       1,     3,    -1,     6,    -1,    81,     6,    -1,   238,    77,
       6,    -1,   238,    77,    81,     6,    -1,    -1,    54,     6,
     240,     3,   241,     9,     3,    -1,    -1,   241,   242,    -1,
       3,    -1,     6,    75,   254,   243,    -1,     6,   243,    -1,
       3,    -1,    77,    -1,    -1,    34,     6,   245,   246,   247,
     237,     9,     3,    -1,   226,     3,    -1,     1,     3,    -1,
      -1,   247,   248,    -1,     3,    -1,   192,    -1,   236,    -1,
     234,    -1,    -1,    36,   250,   251,     3,    -1,   252,    -1,
     251,    77,   252,    -1,   251,    77,     1,    -1,     6,    -1,
      35,     3,    -1,    35,   257,     3,    -1,    35,     1,     3,
      -1,     8,    -1,    55,    -1,    56,    -1,     4,    -1,     5,
      -1,     7,    -1,     6,    -1,   255,    -1,    27,    -1,    26,
      -1,   254,    -1,   256,    -1,    98,   257,    -1,   257,    99,
     257,    -1,   257,    98,   257,    -1,   257,   102,   257,    -1,
     257,   101,   257,    -1,   257,   100,   257,    -1,   257,   103,
     257,    -1,   257,    97,   257,    -1,   257,    96,   257,    -1,
     257,    95,   257,    -1,   257,   105,   257,    -1,   257,   104,
     257,    -1,   106,   257,    -1,   257,    86,   257,    -1,   257,
     111,    -1,   111,   257,    -1,   257,   110,    -1,   110,   257,
      -1,   257,    87,   257,    -1,   257,    85,   257,    -1,   257,
      84,   257,    -1,   257,    83,   257,    -1,   257,    82,   257,
      -1,   257,    80,   257,    -1,   257,    79,   257,    -1,    81,
     257,    -1,   257,    92,   257,    -1,   257,    91,   257,    -1,
     257,    90,   257,    -1,   257,    89,   257,    -1,   257,    88,
       6,    -1,   112,   255,    -1,   112,     4,    -1,    94,   257,
      -1,    93,   257,    -1,   266,    -1,   261,    -1,   264,    -1,
     259,    -1,   269,    -1,   271,    -1,   257,   258,    -1,   270,
      -1,   257,    61,   257,    60,    -1,   257,    61,   102,   257,
      60,    -1,   257,    62,     6,    -1,   272,    -1,   258,    -1,
     257,    75,   257,    -1,   257,    75,   257,    77,   273,    -1,
     257,    74,   257,    -1,   257,    73,   257,    -1,   257,    72,
     257,    -1,   257,    71,   257,    -1,   257,    70,   257,    -1,
     257,    64,   257,    -1,   257,    69,   257,    -1,   257,    68,
     257,    -1,   257,    67,   257,    -1,   257,    65,   257,    -1,
     257,    66,   257,    -1,    59,   257,    58,    -1,    61,    47,
      60,    -1,    61,   257,    47,    60,    -1,    61,    47,   257,
      60,    -1,    61,   257,    47,   257,    60,    -1,    61,   257,
      47,   257,    47,   257,    60,    -1,   257,    59,   273,    58,
      -1,   257,    59,    58,    -1,    -1,   257,    59,   273,     1,
     260,    58,    -1,    -1,    48,   262,   263,   200,   138,     9,
      -1,    59,   198,    58,     3,    -1,    59,   198,     1,    -1,
       1,     3,    -1,    -1,    50,   265,   263,   200,   138,     9,
      -1,    -1,    37,   267,   268,    63,   257,    -1,   198,    -1,
       1,     3,    -1,   257,    78,   257,    47,   257,    -1,   257,
      78,   257,    47,     1,    -1,   257,    78,   257,     1,    -1,
     257,    78,     1,    -1,    61,    60,    -1,    61,   273,    60,
      -1,    61,   273,     1,    -1,    52,    60,    -1,    52,   274,
      60,    -1,    52,   274,     1,    -1,    61,    63,    60,    -1,
      61,   276,    60,    -1,    61,   276,     1,    60,    -1,   257,
      -1,   273,    77,   257,    -1,   257,    -1,   274,   257,    -1,
     255,    -1,   275,    77,   255,    -1,   257,    63,   257,    -1,
     276,    77,   257,    63,   257,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   199,   199,   202,   204,   208,   209,   210,   214,   219,
     220,   225,   230,   235,   240,   241,   242,   243,   247,   248,
     252,   258,   264,   272,   273,   274,   275,   276,   277,   282,
     284,   290,   304,   305,   306,   307,   308,   309,   310,   311,
     312,   313,   314,   315,   316,   317,   318,   319,   320,   321,
     322,   326,   332,   340,   342,   347,   347,   361,   369,   370,
     371,   375,   376,   377,   381,   381,   396,   406,   407,   411,
     412,   416,   418,   419,   419,   428,   429,   434,   434,   446,
     447,   450,   452,   458,   467,   475,   485,   494,   504,   503,
     528,   527,   553,   558,   565,   567,   571,   578,   579,   580,
     584,   597,   605,   609,   615,   621,   628,   633,   642,   652,
     652,   666,   675,   679,   679,   692,   701,   705,   705,   721,
     730,   734,   734,   751,   752,   759,   761,   762,   766,   768,
     767,   778,   778,   790,   790,   802,   802,   818,   821,   820,
     833,   834,   835,   838,   839,   845,   846,   850,   859,   871,
     882,   893,   914,   914,   931,   932,   939,   941,   942,   946,
     948,   947,   958,   958,   971,   971,   983,   983,  1001,  1002,
    1005,  1006,  1018,  1039,  1043,  1048,  1056,  1063,  1062,  1081,
    1082,  1085,  1087,  1091,  1092,  1096,  1101,  1119,  1139,  1149,
    1160,  1168,  1169,  1173,  1185,  1208,  1209,  1216,  1226,  1235,
    1236,  1236,  1240,  1244,  1245,  1245,  1252,  1306,  1308,  1309,
    1313,  1328,  1331,  1330,  1342,  1341,  1356,  1357,  1361,  1362,
    1371,  1375,  1383,  1390,  1405,  1411,  1423,  1433,  1438,  1450,
    1459,  1466,  1474,  1479,  1487,  1492,  1497,  1502,  1516,  1521,
    1526,  1534,  1535,  1539,  1545,  1557,  1562,  1570,  1571,  1575,
    1579,  1583,  1595,  1602,  1612,  1613,  1616,  1617,  1620,  1622,
    1626,  1633,  1634,  1635,  1647,  1646,  1705,  1708,  1714,  1716,
    1717,  1717,  1723,  1725,  1729,  1730,  1734,  1768,  1770,  1779,
    1780,  1784,  1785,  1794,  1797,  1799,  1803,  1804,  1807,  1825,
    1829,  1829,  1863,  1885,  1912,  1914,  1915,  1922,  1930,  1936,
    1942,  1956,  1955,  1999,  2001,  2005,  2006,  2011,  2018,  2018,
    2027,  2026,  2090,  2091,  2097,  2099,  2103,  2104,  2107,  2126,
    2135,  2134,  2152,  2153,  2154,  2161,  2177,  2178,  2179,  2189,
    2190,  2191,  2192,  2193,  2194,  2198,  2216,  2217,  2218,  2229,
    2230,  2231,  2232,  2233,  2234,  2235,  2236,  2237,  2238,  2239,
    2240,  2241,  2242,  2243,  2244,  2245,  2246,  2247,  2248,  2249,
    2250,  2251,  2252,  2253,  2254,  2255,  2256,  2257,  2258,  2259,
    2260,  2261,  2262,  2263,  2264,  2265,  2266,  2267,  2268,  2269,
    2270,  2271,  2273,  2278,  2282,  2287,  2293,  2302,  2303,  2305,
    2310,  2317,  2318,  2319,  2320,  2321,  2322,  2323,  2324,  2325,
    2326,  2327,  2328,  2333,  2336,  2339,  2342,  2345,  2351,  2357,
    2362,  2362,  2372,  2371,  2412,  2413,  2417,  2425,  2424,  2470,
    2469,  2512,  2513,  2522,  2527,  2534,  2541,  2551,  2552,  2556,
    2564,  2565,  2569,  2578,  2579,  2580,  2588,  2589,  2593,  2594,
    2598,  2604,  2611,  2612
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
  "CAP_CAP", "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT",
  "SLASH", "STAR", "POW", "SHR", "SHL", "TILDE", "NEG", "AMPER", "BANG",
  "DECREMENT", "INCREMENT", "DOLLAR", "$accept", "input", "body", "line",
  "toplevel_statement", "INTNUM_WITH_MINUS", "load_statement", "statement",
  "base_statement", "assignment_def_list", "def_statement",
  "while_statement", "@1", "while_decl", "while_short_decl",
  "if_statement", "@2", "if_decl", "if_short_decl", "elif_or_else", "@3",
  "else_decl", "elif_statement", "@4", "elif_decl", "statement_list",
  "break_statement", "continue_statement", "forin_statement", "@5", "@6",
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
  "import_statement", "symbol_or_string", "import_symbol_list",
  "directive_statement", "directive_pair_list", "directive_pair",
  "attributes_statement", "attributes_decl", "attributes_short_decl",
  "attribute_list", "attribute_vert_list", "class_decl", "@26",
  "class_def_inner", "class_param_list", "@27", "from_clause",
  "inherit_list", "inherit_token", "inherit_call", "inherit_param_list",
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
     365,   366,   367
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   113,   114,   115,   115,   116,   116,   116,   117,   117,
     117,   117,   117,   117,   117,   117,   117,   117,   118,   118,
     119,   119,   119,   120,   120,   120,   120,   120,   120,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   121,   121,   121,   121,   121,   121,   121,   121,   121,
     121,   122,   122,   123,   123,   125,   124,   124,   126,   126,
     126,   127,   127,   127,   129,   128,   128,   130,   130,   131,
     131,   132,   132,   133,   132,   134,   134,   136,   135,   137,
     137,   138,   138,   139,   139,   140,   140,   140,   142,   141,
     143,   141,   141,   141,   144,   144,   145,   145,   145,   145,
     146,   146,   147,   147,   147,   147,   147,   147,   148,   150,
     149,   149,   149,   152,   151,   151,   151,   154,   153,   153,
     153,   156,   155,   157,   157,   158,   158,   158,   159,   160,
     159,   161,   159,   162,   159,   163,   159,   164,   165,   164,
     166,   166,   166,   167,   167,   168,   168,   169,   169,   169,
     169,   169,   171,   170,   172,   172,   173,   173,   173,   174,
     175,   174,   176,   174,   177,   174,   178,   174,   179,   179,
     180,   180,   180,   181,   181,   181,   182,   183,   182,   184,
     184,   185,   185,   186,   186,   187,   188,   188,   188,   188,
     188,   189,   189,   190,   190,   191,   191,   192,   192,   193,
     194,   193,   193,   195,   196,   195,   197,   198,   198,   198,
     199,   200,   201,   200,   202,   200,   203,   203,   204,   204,
     205,   205,   206,   206,   206,   206,   207,   207,   207,   208,
     208,   208,   209,   209,   210,   210,   210,   210,   210,   210,
     210,   211,   211,   212,   212,   213,   213,   214,   214,   215,
     215,   215,   216,   216,   217,   217,   218,   218,   219,   219,
     219,   220,   220,   220,   222,   221,   223,   223,   224,   224,
     225,   224,   226,   226,   227,   227,   228,   229,   229,   230,
     230,   231,   231,   231,   232,   232,   233,   233,   233,   233,
     235,   234,   236,   236,   237,   237,   237,   238,   238,   238,
     238,   240,   239,   241,   241,   242,   242,   242,   243,   243,
     245,   244,   246,   246,   247,   247,   248,   248,   248,   248,
     250,   249,   251,   251,   251,   252,   253,   253,   253,   254,
     254,   254,   254,   254,   254,   255,   256,   256,   256,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   258,   258,   258,   258,   258,   259,   259,
     260,   259,   262,   261,   263,   263,   263,   265,   264,   267,
     266,   268,   268,   269,   269,   269,   269,   270,   270,   270,
     271,   271,   271,   272,   272,   272,   273,   273,   274,   274,
     275,   275,   276,   276
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
       3,     3,     1,     3,     3,     5,     7,     4,     4,     6,
       3,     1,     1,     1,     3,     3,     3,     1,     3,     3,
       3,     3,     4,     3,     2,     3,     2,     3,     0,     1,
       3,     2,     3,     2,     0,     8,     3,     2,     0,     3,
       0,     5,     0,     2,     1,     3,     2,     0,     3,     1,
       3,     1,     1,     1,     0,     2,     1,     1,     1,     1,
       0,     7,     5,     4,     0,     3,     3,     1,     2,     3,
       4,     0,     7,     0,     2,     1,     4,     2,     1,     1,
       0,     8,     2,     2,     0,     2,     1,     1,     1,     1,
       0,     4,     1,     3,     3,     1,     2,     3,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     2,     2,     2,     2,     3,
       3,     3,     3,     3,     3,     3,     2,     3,     3,     3,
       3,     3,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     2,     1,     4,     5,     3,     1,     1,     3,
       5,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     4,     4,     5,     7,     4,     3,
       0,     6,     0,     6,     4,     3,     2,     0,     6,     0,
       5,     1,     2,     5,     5,     4,     3,     2,     3,     3,
       2,     3,     3,     3,     3,     4,     1,     3,     1,     2,
       1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   332,   333,   335,   334,
     329,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   338,   337,     0,     0,     0,     0,     0,     0,   320,
     419,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     417,     0,     0,    59,     0,   330,   331,   108,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       4,     5,     8,    13,    23,    32,    33,    55,     0,    37,
      64,     0,    38,    39,    34,    47,    48,    49,    35,   121,
      36,   152,    40,    41,   177,    42,    10,   211,     0,     0,
      45,    46,    14,    15,    16,     9,    17,     0,   258,    11,
      50,    12,    44,    43,   339,   336,   340,   436,   388,   379,
     377,   378,   376,   380,   383,   381,   387,     0,    28,     6,
       0,     0,     0,     0,   412,     0,     0,    83,     0,    85,
       0,     0,     0,     0,   440,     0,     0,     0,     0,     0,
       0,     0,   436,     0,     0,   179,     0,     0,     0,     0,
     264,     0,   310,     0,   326,     0,     0,     0,     0,     0,
       0,     0,     0,   379,     0,     0,     0,   254,   256,     0,
       0,     0,   229,   232,     0,     0,   243,     0,     0,     0,
       0,     0,   247,     0,   206,     0,     0,     0,     0,   430,
     438,     0,    62,   301,     0,     0,   427,     0,   436,     0,
       0,   366,     0,   105,     0,   375,   374,   341,     0,   103,
       0,   353,   358,   356,   373,   372,    81,     0,     0,     0,
      57,    81,    66,   125,   156,    81,     0,    81,   212,   214,
     198,     0,     0,     0,   259,     0,   258,     0,    29,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   357,
     355,   382,     0,     0,    54,    53,     0,     0,    60,    63,
      58,    61,    84,    87,    86,    68,    70,    67,    69,    93,
       0,     0,   124,   123,     7,   155,   154,   175,     0,     0,
     180,   176,   196,   195,    27,     0,    26,     0,   328,   327,
     325,     0,   322,     0,   210,   421,   208,     0,    22,    20,
      21,   221,   220,   228,     0,   255,   257,   225,   222,     0,
     231,   230,     0,   240,   241,   242,     0,     0,   234,     0,
       0,   246,     0,   245,     0,    25,     0,   207,   211,   211,
     101,   100,   432,   431,   439,     0,   402,   403,     0,   433,
       0,     0,   429,   428,     0,   434,     0,   107,   104,   106,
     102,     0,     0,     0,     0,     0,     0,   216,   218,     0,
      81,     0,   202,   204,     0,   263,   261,     0,     0,     0,
     253,   409,     0,     0,     0,   386,   396,   400,   401,   399,
     398,   397,   395,   394,   393,   392,   391,   389,   426,     0,
     365,   364,   363,   362,   361,   360,   354,   359,   371,   370,
     369,   368,   367,   350,   349,   348,   343,   342,   346,   345,
     344,   347,   352,   351,     0,   437,     0,    51,   441,     0,
       0,   174,     0,     0,   207,   284,   272,     0,     0,     0,
     314,   321,     0,   422,     0,     0,     0,     0,     0,   369,
     233,   238,     0,   237,     0,   244,    18,   249,   250,     0,
     251,   248,   416,     0,    81,    81,   303,   405,   404,     0,
     442,   435,     0,     0,    82,     0,     0,     0,    73,    72,
      77,     0,   128,     0,     0,   126,     0,   138,     0,   159,
       0,     0,   157,     0,     0,   182,   183,    81,   217,   219,
       0,     0,   215,     0,   200,     0,   260,   252,   262,   410,
     408,     0,   384,     0,   425,     0,    30,     0,     0,    92,
      88,    90,   173,   267,     0,   294,     0,   313,   277,   273,
     274,   312,   294,   324,   323,   209,   420,   227,   226,   224,
     223,     0,   235,     0,    19,   415,     0,     0,     0,     0,
       0,   406,     0,    56,     0,    75,     0,     0,     0,    81,
      81,   127,     0,   151,   149,   147,   148,     0,   145,   142,
       0,     0,   158,     0,   171,   172,     0,   168,     0,     0,
     186,   193,   194,     0,     0,   191,     0,   184,     0,   197,
       0,     0,     0,   199,   203,     0,   385,   390,   424,   423,
       0,    52,     0,     0,   270,   269,   286,     0,     0,     0,
       0,     0,   287,   285,   289,   288,     0,   266,     0,   276,
       0,   316,   317,   319,   318,     0,   315,   239,     0,   414,
     413,   418,   305,     0,     0,   304,     0,   443,    76,    80,
      79,    65,     0,     0,   133,   135,     0,   129,   131,     0,
     122,    81,     0,   139,   164,   166,   160,   162,   170,   153,
     190,     0,   188,     0,     0,   178,   213,   205,     0,   411,
      31,     0,     0,     0,    96,     0,     0,    97,    98,    99,
      91,     0,     0,   290,     0,     0,   297,     0,     0,     0,
     282,   283,     0,   279,   281,   275,     0,   236,   308,     0,
     309,   307,   302,   407,    78,    81,     0,   150,    81,     0,
     146,     0,   144,    81,     0,    81,     0,   169,   187,   192,
       0,   201,     0,   109,     0,     0,   113,     0,     0,   117,
       0,     0,    95,   271,     0,   211,     0,   296,   298,   295,
       0,   265,   278,     0,   311,     0,     0,   136,     0,   132,
       0,   167,     0,   163,   189,   112,    81,   111,   116,    81,
     115,   120,    81,   119,    89,   293,    81,     0,   299,     0,
     280,   306,     0,     0,     0,     0,   292,   300,     0,     0,
       0,     0,   110,   114,   118,   291
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    60,    61,   586,    62,   494,    64,   121,
      65,    66,   216,    67,    68,    69,   221,    70,    71,   497,
     579,   498,   499,   580,   500,   381,    72,    73,    74,   622,
     623,   695,   696,    75,    76,    77,   697,   776,   698,   779,
     699,   782,    78,   223,    79,   383,   505,   728,   729,   725,
     726,   506,   591,   507,   673,   587,   588,    80,   224,    81,
     384,   512,   735,   736,   733,   734,   596,   597,    82,    83,
     225,    84,   514,   515,   516,   517,   604,   605,    85,    86,
      87,   612,    88,   523,    89,   325,   326,   227,   390,   391,
     228,   229,    90,    91,    92,    93,   174,    94,   346,   178,
      95,   181,   182,    96,    97,    98,   235,   236,    99,   315,
     455,   456,   701,   459,   549,   550,   639,   712,   713,   545,
     633,   634,   755,   635,   636,   708,   100,   365,   569,   655,
     721,   101,   317,   460,   552,   646,   102,   156,   321,   322,
     103,   104,   105,   106,   107,   108,   109,   615,   110,   185,
     358,   111,   186,   112,   157,   327,   113,   114,   115,   116,
     117,   191,   135,   200
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -348
static const yytype_int16 yypact[] =
{
    -348,    98,   759,  -348,    28,  -348,  -348,  -348,  -348,  -348,
    -348,   110,   339,  3194,   309,   448,  3259,   343,  3324,    84,
    3389,  -348,  -348,  3454,   314,  3519,   433,   502,   463,  -348,
    -348,   372,  3584,   565,   322,  3649,   561,   476,   571,   116,
    -348,  3714,   274,    78,   188,  -348,  -348,  -348,  5258,  4966,
    5258,   575,  5258,  5258,  5258,  3064,  5258,  5258,  5258,   347,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  2999,  -348,
    -348,  2999,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,   166,  2999,   118,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,    59,   220,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  4381,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,   379,  -348,  -348,
     280,   111,   202,    25,  -348,  4155,   290,  -348,   300,  -348,
     310,   106,  4215,   311,  -348,    88,   317,  4434,   325,   349,
    4487,   367,  5749,    20,   368,  -348,  2999,   371,  4540,   374,
    -348,   378,  -348,   401,  -348,  4593,   298,    75,   437,   444,
     445,   454,  5749,   471,   485,   307,   291,  -348,  -348,   488,
    4646,   496,  -348,  -348,   112,   503,  -348,   179,    31,   517,
     366,   124,  -348,   540,  -348,   131,   131,   548,  4699,  -348,
    5749,  3129,  -348,  -348,  5484,  5063,  -348,   467,  5312,    86,
      89,  5984,   556,  -348,   152,   436,   436,   226,   562,  -348,
     214,   226,   226,   226,  -348,  -348,  -348,   583,   584,   136,
    -348,  -348,  -348,  -348,  -348,  -348,   460,  -348,  -348,  -348,
    -348,   589,    73,   593,  -348,   215,   155,   218,  -348,  5128,
    4993,   582,  5258,  5258,  5258,  5258,  5258,  5258,  5258,  5258,
    5258,  5258,  5258,  5258,  3779,  5258,  5258,  5258,  5258,  5258,
    5258,  5258,  5258,   598,  5258,  5258,  5258,  5258,  5258,  5258,
    5258,  5258,  5258,  5258,  5258,  5258,  5258,  5258,  5258,  -348,
    -348,  -348,  5258,  5258,  -348,  -348,   600,  5258,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
     600,  3844,  -348,  -348,  -348,  -348,  -348,  -348,   596,  5258,
    -348,  -348,  -348,  -348,  -348,   283,  -348,    91,  -348,  -348,
    -348,   219,  -348,   606,  -348,   536,  -348,   552,  -348,  -348,
    -348,  -348,  -348,  -348,   301,  -348,  -348,  -348,  -348,  3909,
    -348,  -348,   610,  -348,  -348,  -348,    85,   616,  -348,   179,
     620,  -348,     8,  -348,   622,  -348,   619,   623,   166,   166,
    -348,  -348,  -348,  -348,  5749,   621,  -348,  -348,  5537,  -348,
    5193,  5258,  -348,  -348,   577,  -348,  5258,  -348,  -348,  -348,
    -348,  1767,  1431,   449,   594,  1543,   315,  -348,  -348,  1879,
    -348,  2999,  -348,  -348,    26,  -348,  -348,   629,   637,   222,
    -348,  -348,    81,  5258,  5371,  -348,  5749,  5749,  5749,  5749,
    5749,  5749,  5749,  5749,  5749,  5749,  5749,  5796,  -348,  4095,
    5937,  5984,  6031,  6031,  6031,  6031,  6031,  6031,  -348,   436,
     436,   436,   436,   618,   618,   549,  1393,  1393,   376,   376,
     376,   528,   226,   226,  4268,  5843,   570,  5749,  -348,   652,
    4328,  -348,   264,   654,   623,  -348,   625,   658,   656,   660,
    -348,  -348,   585,  -348,   623,  5258,   661,   662,   663,   109,
    -348,  -348,   665,  -348,   107,  -348,  -348,  -348,  -348,   668,
    -348,  -348,  -348,    82,  -348,  -348,  -348,  -348,  -348,  5430,
    5749,  -348,  5590,   664,  -348,   509,  3974,   666,  -348,  -348,
    -348,   673,  -348,    18,   482,  -348,   669,  -348,   679,  -348,
      74,   674,  -348,    67,   675,   659,  -348,  -348,  -348,  -348,
     685,  1991,  -348,   633,  -348,   328,  -348,  -348,  -348,  -348,
    -348,  5643,  -348,  5258,  -348,  4039,  -348,  5258,  5258,  -348,
    -348,  -348,  -348,  -348,   104,    51,   689,  -348,   634,   624,
    -348,  -348,    53,  -348,  -348,  -348,  5749,  -348,  -348,  -348,
    -348,   692,  -348,   690,  -348,  -348,   694,  2103,  2215,   357,
    5258,  -348,  5258,  -348,   695,  -348,   696,  4752,   699,  -348,
    -348,  -348,   329,  -348,  -348,  -348,   628,   126,  -348,  -348,
     702,   336,  -348,   340,  -348,  -348,   221,  -348,   703,   704,
    -348,  -348,  -348,   600,    30,  -348,   706,  -348,  1655,  -348,
     708,   667,   655,  -348,  -348,   672,  -348,   635,  -348,  5890,
     269,  5749,   871,  2999,  -348,  -348,  -348,   650,   723,   725,
     727,    52,  -348,  -348,  -348,  -348,   726,  -348,   438,  -348,
     656,  -348,  -348,  -348,  -348,   728,  -348,  -348,   733,  -348,
    -348,  -348,  -348,    92,   735,  -348,  5696,  5749,  -348,  -348,
    -348,  -348,  2327,  1431,  -348,  -348,     6,  -348,  -348,    44,
    -348,  -348,  2999,  -348,  -348,  -348,  -348,  -348,   510,  -348,
    -348,   736,  -348,   524,   600,  -348,  -348,  -348,   738,  -348,
    -348,   547,   567,   597,  -348,   734,   871,  -348,  -348,  -348,
    -348,   688,  5258,  -348,   677,   744,  -348,   743,   270,   747,
    -348,  -348,    77,  -348,  -348,  -348,   750,  -348,  -348,   497,
    -348,  -348,  -348,  -348,  -348,  -348,  2999,  -348,  -348,  2999,
    -348,  2439,  -348,  -348,  2999,  -348,  2999,  -348,  -348,  -348,
     752,  -348,   753,  -348,  2999,   755,  -348,  2999,   772,  -348,
    2999,   773,  -348,  -348,  4805,   166,  5258,  -348,  -348,  -348,
      49,  -348,  -348,   438,  -348,   271,   983,  -348,  1095,  -348,
    1207,  -348,  1319,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  4858,  -348,   774,
    -348,  -348,  2551,  2663,  2775,  2887,  -348,  -348,   775,   776,
     780,   786,  -348,  -348,  -348,  -348
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -348,  -348,  -348,  -348,  -348,  -347,  -348,    -2,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,   129,
    -348,  -348,  -348,  -348,  -348,   -25,  -348,  -348,  -348,  -348,
    -348,   101,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
    -348,   422,  -348,  -348,  -348,  -348,   139,  -348,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,   141,  -348,  -348,
    -348,  -348,  -348,  -348,   302,  -348,  -348,   138,  -348,  -329,
    -348,  -348,  -348,  -348,  -348,  -185,   358,  -341,  -348,  -348,
    -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,   474,  -348,
    -348,  -348,   470,  -348,  -348,  -348,   -89,  -348,  -348,  -348,
    -348,  -348,  -348,   369,  -348,   186,  -348,  -348,    64,  -348,
    -348,   276,  -348,   277,   278,  -348,  -348,  -348,  -348,  -348,
      66,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,   370,
    -348,  -331,   -10,  -348,   -12,   -14,   801,  -348,  -348,  -348,
     648,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,  -348,
      12,  -348,  -348,  -348
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -413
static const yytype_int16 yytable[] =
{
      63,   125,   122,   467,   132,   480,   137,   134,   140,   237,
     476,   142,   476,   148,   477,   478,   155,   484,   485,   582,
     162,   308,   476,   170,   583,   584,   585,   524,   288,   188,
     190,   118,   347,   682,   348,   143,   194,   198,   201,   142,
     205,   206,   207,   142,   211,   212,   213,   394,   476,   215,
     583,   584,   585,   705,   626,   788,   641,   627,   706,   627,
     233,   199,  -258,   204,   349,   234,   220,   210,   599,   222,
     600,   601,   289,   602,   393,   593,   323,  -170,   594,   324,
     595,   324,   529,   565,   525,   138,   230,   372,   471,   628,
     374,   628,   457,   281,  -272,   718,   309,   283,     3,   629,
     630,   629,   630,   464,   479,   624,   479,   683,   350,   295,
     562,   281,   560,   119,   285,   341,   479,   183,   281,   231,
     684,  -170,   184,   281,   458,   192,   281,   353,   281,   667,
     789,  -207,   356,   707,   281,   762,  -258,   183,  -207,   530,
     566,   281,   479,   631,   311,   631,   373,   399,   281,   375,
    -207,  -170,  -207,   296,   763,   378,   281,   603,   283,   464,
     472,   234,   625,   283,   398,   300,   376,   719,   239,   720,
     240,   241,   483,   668,   281,  -412,   281,   232,   301,   364,
     281,   464,   563,   368,   281,   344,   345,   281,   286,   342,
     357,   281,   281,   281,   193,  -412,   382,   281,   281,   281,
     385,   354,   389,   669,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   226,   632,   380,   396,   279,
     280,   400,   461,   642,   676,   528,   234,   142,   404,   283,
     406,   407,   408,   409,   410,   411,   412,   413,   414,   415,
     416,   417,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   402,   429,   430,   431,   432,   433,   434,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   542,   677,   544,
     444,   445,   690,   759,   718,   447,   446,   287,     6,     7,
       8,     9,    10,   284,   453,   239,  -268,   240,   241,   450,
     448,   283,   397,   292,   335,   397,   462,   142,   678,   397,
      21,    22,   466,   293,   320,     6,     7,   714,     9,    10,
     126,    30,   127,   294,   299,   144,  -268,   145,   518,   727,
     302,   452,   124,   166,    40,   167,    42,   469,   304,    45,
      46,   613,   664,    48,   189,    49,   279,   280,   336,   671,
     120,   283,   454,   674,   133,     8,   283,   760,   720,     8,
     281,   214,   305,     8,   281,    50,    45,    46,   489,   490,
     652,   146,   519,   653,   492,   521,   654,    52,    53,   168,
     307,   310,    54,   158,   312,   614,   665,   314,   159,   160,
      56,   316,   334,   672,    57,    58,    59,   675,   765,   522,
     281,   531,   281,   281,   281,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   318,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   786,   281,   281,   281,   281,   281,
     281,   281,   281,   281,   281,   281,   281,   281,   281,   281,
     281,   281,   714,   281,   149,   239,   281,   240,   241,   150,
     328,   352,     6,     7,   710,     9,    10,   329,   330,   128,
     501,   129,   502,   556,   282,   281,   283,   331,  -137,   567,
     568,   386,   130,   387,   153,   711,   154,     6,     7,     8,
       9,    10,   503,   504,   332,   281,   281,   175,   281,   276,
     277,   278,   176,   589,   577,  -141,   279,   280,   333,    21,
      22,   337,   608,    45,    46,   239,  -140,   240,   241,   340,
      30,     6,     7,   151,     9,    10,   343,   388,   152,   177,
     574,   124,   575,    40,   594,    42,   595,   281,    45,    46,
     351,   142,    48,   619,    49,   142,   621,   369,   601,  -141,
     602,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   281,   355,    50,   617,   279,   280,   742,   620,
     743,   360,    45,    46,   662,   663,    52,    53,   656,   377,
     657,    54,   171,   281,   172,   379,   164,   173,   745,    56,
     746,   165,   179,    57,    58,    59,   202,   180,   203,     6,
       7,     8,     9,    10,   149,   151,   553,   239,   405,   240,
     241,   320,   392,   681,   744,   508,   395,   509,   748,   451,
     749,    21,    22,  -137,   428,   281,     8,   281,   239,   463,
     240,   241,    30,   464,   747,   465,   470,   510,   504,   473,
     694,   700,   482,   124,   486,    40,   475,    42,   180,   324,
      45,    46,   277,   278,    48,   526,    49,   491,   279,   280,
     527,  -140,   281,   281,   750,   538,   731,   271,   272,   273,
     274,   275,   276,   277,   278,   539,    50,   543,   458,   279,
     280,   547,   548,   551,   557,   558,   559,   573,    52,    53,
     732,   561,   564,    54,   740,   578,   581,   239,   590,   240,
     241,    56,   592,   598,   606,    57,    58,    59,   609,   513,
     754,   611,   637,   638,   694,   647,   648,   649,   658,   659,
     766,   640,   661,   768,   666,   670,   679,   680,   770,   685,
     772,   686,   283,   688,   687,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   767,   702,   703,   769,   279,   280,
     689,   184,   771,   704,   773,   709,   717,   716,   722,   738,
     281,   741,   777,   751,   787,   780,   753,   757,   783,   758,
     761,   792,   756,   764,   793,   774,   775,   794,   778,    -2,
       4,   795,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,   281,    16,   781,   784,    17,   802,   803,
     797,    18,    19,   804,    20,    21,    22,    23,    24,   805,
      25,    26,   724,    27,    28,    29,    30,   752,    31,    32,
      33,    34,    35,    36,    37,    38,   511,    39,   730,    40,
      41,    42,    43,    44,    45,    46,    47,   607,    48,   737,
      49,   739,   555,   474,   481,   546,   715,   790,   643,   644,
     645,   791,   554,   163,   359,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,     0,     0,    57,
      58,    59,     4,     0,     5,     6,     7,     8,     9,    10,
     -94,    12,    13,    14,    15,     0,    16,     0,     0,    17,
     691,   692,   693,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   217,     0,   218,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   219,
       0,    40,    41,    42,    43,    44,    45,    46,    47,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,     0,
       0,    57,    58,    59,     4,     0,     5,     6,     7,     8,
       9,    10,  -134,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,  -134,  -134,    20,    21,
      22,    23,    24,     0,    25,   217,     0,   218,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
    -134,   219,     0,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,     0,     0,    57,    58,    59,     4,     0,     5,     6,
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
       0,    56,     0,     0,     0,    57,    58,    59,     4,     0,
       5,     6,     7,     8,     9,    10,  -165,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
    -165,  -165,    20,    21,    22,    23,    24,     0,    25,   217,
       0,   218,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,  -165,   219,     0,    40,    41,    42,
      43,    44,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,     0,     0,    57,    58,    59,
       4,     0,     5,     6,     7,     8,     9,    10,  -161,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,  -161,  -161,    20,    21,    22,    23,    24,     0,
      25,   217,     0,   218,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,  -161,   219,     0,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,     0,     0,    57,
      58,    59,     4,     0,     5,     6,     7,     8,     9,    10,
     -71,    12,    13,    14,    15,     0,    16,   495,   496,    17,
       0,     0,   239,    18,   240,   241,    20,    21,    22,    23,
      24,     0,    25,   217,     0,   218,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   219,
       0,    40,    41,    42,    43,    44,    45,    46,    47,     0,
      48,     0,    49,   273,   274,   275,   276,   277,   278,     0,
       0,     0,     0,   279,   280,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,     0,
       0,    57,    58,    59,     4,     0,     5,     6,     7,     8,
       9,    10,  -181,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,   513,    25,   217,     0,   218,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   219,     0,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,     0,     0,    57,    58,    59,     4,     0,     5,     6,
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
       0,    56,     0,     0,     0,    57,    58,    59,     4,     0,
       5,     6,     7,     8,     9,    10,   493,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   217,
       0,   218,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   219,     0,    40,    41,    42,
      43,    44,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,     0,     0,    57,    58,    59,
       4,     0,     5,     6,     7,     8,     9,    10,   520,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   217,     0,   218,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   219,     0,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,     0,     0,    57,
      58,    59,     4,     0,     5,     6,     7,     8,     9,    10,
     610,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   217,     0,   218,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   219,
       0,    40,    41,    42,    43,    44,    45,    46,    47,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,     0,
       0,    57,    58,    59,     4,     0,     5,     6,     7,     8,
       9,    10,   650,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   217,     0,   218,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   219,     0,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,     0,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   651,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,     0,     0,    57,    58,    59,     4,     0,
       5,     6,     7,     8,     9,    10,   -74,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   217,
       0,   218,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   219,     0,    40,    41,    42,
      43,    44,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,     0,     0,    57,    58,    59,
       4,     0,     5,     6,     7,     8,     9,    10,  -143,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   217,     0,   218,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   219,     0,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,     0,    55,     0,    56,     0,     0,     0,    57,
      58,    59,     4,     0,     5,     6,     7,     8,     9,    10,
     798,    12,    13,    14,    15,     0,    16,     0,     0,    17,
       0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
      24,     0,    25,   217,     0,   218,    28,    29,    30,     0,
       0,    32,     0,     0,    35,     0,     0,     0,     0,   219,
       0,    40,    41,    42,    43,    44,    45,    46,    47,     0,
      48,     0,    49,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,     0,     0,    52,    53,     0,     0,     0,    54,
       0,     0,     0,     0,     0,    55,     0,    56,     0,     0,
       0,    57,    58,    59,     4,     0,     5,     6,     7,     8,
       9,    10,   799,    12,    13,    14,    15,     0,    16,     0,
       0,    17,     0,     0,     0,    18,     0,     0,    20,    21,
      22,    23,    24,     0,    25,   217,     0,   218,    28,    29,
      30,     0,     0,    32,     0,     0,    35,     0,     0,     0,
       0,   219,     0,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,     0,
       0,    54,     0,     0,     0,     0,     0,    55,     0,    56,
       0,     0,     0,    57,    58,    59,     4,     0,     5,     6,
       7,     8,     9,    10,   800,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,    24,     0,    25,   217,     0,   218,
      28,    29,    30,     0,     0,    32,     0,     0,    35,     0,
       0,     0,     0,   219,     0,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,     0,     0,    52,    53,
       0,     0,     0,    54,     0,     0,     0,     0,     0,    55,
       0,    56,     0,     0,     0,    57,    58,    59,     4,     0,
       5,     6,     7,     8,     9,    10,   801,    12,    13,    14,
      15,     0,    16,     0,     0,    17,     0,     0,     0,    18,
       0,     0,    20,    21,    22,    23,    24,     0,    25,   217,
       0,   218,    28,    29,    30,     0,     0,    32,     0,     0,
      35,     0,     0,     0,     0,   219,     0,    40,    41,    42,
      43,    44,    45,    46,    47,     0,    48,     0,    49,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,     0,
      52,    53,     0,     0,     0,    54,     0,     0,     0,     0,
       0,    55,     0,    56,     0,     0,     0,    57,    58,    59,
       4,     0,     5,     6,     7,     8,     9,    10,     0,    12,
      13,    14,    15,     0,    16,     0,     0,    17,     0,     0,
       0,    18,     0,     0,    20,    21,    22,    23,    24,     0,
      25,   217,     0,   218,    28,    29,    30,     0,     0,    32,
       0,     0,    35,     0,     0,     0,     0,   219,     0,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,     0,     0,     0,     0,   208,     0,   209,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,    55,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     362,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,   363,
      49,     0,     0,     0,     0,   123,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     131,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   136,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     139,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   141,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     147,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   161,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     169,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   187,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     418,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   449,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     468,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,     0,     0,    48,     0,
      49,     0,     0,     0,     0,   576,     0,     0,     6,     7,
       8,     9,    10,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      21,    22,    52,    53,     0,     0,     0,    54,     0,     0,
       0,    30,     0,     0,     0,    56,     0,     0,     0,    57,
      58,    59,   124,     0,    40,     0,    42,     0,     0,    45,
      46,     0,     0,    48,     0,    49,     0,     0,     0,     0,
     618,     0,     0,     6,     7,     8,     9,    10,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    21,    22,    52,    53,     0,
       0,     0,    54,     0,     0,     0,    30,     0,     0,     0,
      56,     0,     0,     0,    57,    58,    59,   124,     0,    40,
       0,    42,     0,     0,    45,    46,   534,     0,    48,     0,
      49,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    52,    53,     0,     0,     0,    54,     0,     0,
       0,     0,   535,     0,     0,    56,     0,     0,     0,    57,
      58,    59,     0,     0,   239,     0,   240,   241,   290,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,   291,     0,     0,   279,   280,     0,     0,     0,
       0,     0,     0,     0,   239,     0,   240,   241,   297,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,   298,     0,     0,   279,   280,     0,     0,     0,
       0,   536,     0,     0,   239,     0,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,     0,   279,   280,   239,     0,   240,
     241,   540,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,   537,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,   541,     0,     0,   279,   280,
       0,     0,     0,     0,   238,     0,     0,   239,     0,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,     0,   303,   279,   280,
     239,     0,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,     0,
     306,   279,   280,   239,     0,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,     0,   313,   279,   280,   239,     0,   240,   241,
       0,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,     0,     0,   254,   255,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,     0,   319,   279,   280,   239,
       0,   240,   241,     0,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,     0,   338,
     279,   280,   239,     0,   240,   241,     0,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,     0,   361,   279,   280,   239,     0,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   339,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,     0,   660,   279,   280,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,     0,     0,     0,   785,   279,
     280,   239,     0,   240,   241,     0,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,     0,     0,
       0,   796,   279,   280,   239,     0,   240,   241,     0,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
     253,     0,     0,   254,   255,   256,     0,   257,   258,   259,
     260,   261,   262,   263,   264,   265,   266,   267,     0,     0,
     268,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,     0,     0,     0,     0,   279,   280,   239,     0,   240,
     241,     0,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,     0,     0,   254,   255,   256,     0,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,     0,     0,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,     0,     0,     0,     0,   279,   280,
       6,     7,     8,     9,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    21,    22,     0,     0,     0,     6,     7,     8,
       9,    10,     0,    30,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   195,   124,     0,    40,     0,    42,    21,
      22,    45,    46,     0,     0,    48,   196,    49,     0,   197,
      30,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     195,   124,     0,    40,     0,    42,     0,    50,    45,    46,
       0,     0,    48,     0,    49,     0,     0,     0,     0,    52,
      53,     0,     0,     0,    54,     0,     0,     6,     7,     8,
       9,    10,    56,     0,    50,     0,    57,    58,    59,     0,
       0,     0,     0,     0,     0,     0,    52,    53,     0,    21,
      22,    54,     0,     0,     0,   403,     0,     0,     0,    56,
      30,     0,     0,    57,    58,    59,     0,     0,     0,     0,
       0,   124,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,   367,    49,     0,     0,     0,     0,     0,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,     0,     0,    57,    58,    59,   124,     0,    40,     0,
      42,     0,     0,    45,    46,     0,   401,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       9,    10,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,    52,    53,     0,     0,     0,    54,     0,     0,     0,
      30,     0,     0,     0,    56,     0,     0,     0,    57,    58,
      59,   124,     0,    40,     0,    42,     0,     0,    45,    46,
       0,     0,    48,   488,    49,     0,     0,     0,     0,     0,
       0,     0,     6,     7,     8,     9,    10,     0,     0,     0,
       0,     0,     0,     0,    50,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,    22,    52,    53,     0,     0,
       0,    54,     0,     0,     0,    30,     0,     0,     0,    56,
       0,     0,     0,    57,    58,    59,   124,     0,    40,     0,
      42,     0,     0,    45,    46,     0,     0,    48,     0,    49,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    52,    53,     0,     0,     0,    54,     0,     0,   370,
       0,     0,     0,     0,    56,     0,     0,     0,    57,    58,
      59,   239,     0,   240,   241,   371,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,   253,     0,     0,
     254,   255,   256,     0,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   266,   267,     0,     0,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   370,     0,
       0,     0,   279,   280,     0,     0,     0,     0,     0,     0,
     239,   532,   240,   241,     0,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,   253,     0,     0,   254,
     255,   256,     0,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   570,     0,     0,
       0,   279,   280,     0,     0,     0,     0,     0,     0,   239,
     571,   240,   241,     0,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,     0,     0,
     279,   280,   366,   239,     0,   240,   241,     0,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,   253,
       0,     0,   254,   255,   256,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
       0,     0,     0,     0,   279,   280,   239,   487,   240,   241,
       0,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,     0,     0,   254,   255,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,     0,     0,     0,     0,   279,   280,   239,
       0,   240,   241,   572,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,     0,     0,   254,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,     0,     0,     0,     0,
     279,   280,   239,   616,   240,   241,     0,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,     0,
       0,     0,     0,   279,   280,   239,   723,   240,   241,     0,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,     0,     0,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,     0,     0,     0,     0,   279,   280,   239,     0,
     240,   241,     0,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,     0,     0,   254,   255,   256,
       0,   257,   258,   259,   260,   261,   262,   263,   264,   265,
     266,   267,     0,     0,   268,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   239,     0,   240,   241,   279,
     280,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   253,     0,   533,   254,   255,   256,     0,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,     0,
       0,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   239,     0,   240,   241,   279,   280,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   254,   255,   256,     0,   257,   258,   259,   260,   261,
     262,   263,   264,   265,   266,   267,     0,     0,   268,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   239,
       0,   240,   241,   279,   280,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   255,
     256,     0,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,     0,     0,   268,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   239,     0,   240,   241,
     279,   280,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   256,     0,   257,
     258,   259,   260,   261,   262,   263,   264,   265,   266,   267,
       0,     0,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   239,     0,   240,   241,   279,   280,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   257,   258,   259,   260,
     261,   262,   263,   264,   265,   266,   267,     0,     0,   268,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     239,     0,   240,   241,   279,   280,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   263,
     264,   265,   266,   267,     0,     0,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     0,     0,     0,
       0,   279,   280
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   334,    16,   352,    18,    17,    20,    98,
       4,    23,     4,    25,     6,     7,    28,   358,   359,     1,
      32,     1,     4,    35,     6,     7,     8,     1,     3,    41,
      42,     3,     1,     3,     3,    23,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,   232,     4,    59,
       6,     7,     8,     1,     3,     6,     3,     6,     6,     6,
       1,    49,     3,    51,    33,     6,    68,    55,     1,    71,
       3,     4,    47,     6,     1,     1,     1,     3,     4,     6,
       6,     6,     1,     1,    58,     1,    88,     1,     3,    38,
       1,    38,     1,   107,     3,     3,    76,    77,     0,    48,
      49,    48,    49,    77,    98,     1,    98,    77,    77,     3,
       3,   125,     3,     3,     3,     3,    98,     1,   132,     1,
      90,    47,     6,   137,    33,    47,   140,     3,   142,     3,
      81,    58,     1,    81,   148,    58,    77,     1,    63,    58,
      58,   155,    98,    92,   146,    92,    60,   236,   162,    60,
      77,    77,    77,    47,    77,     3,   170,    90,    77,    77,
      75,     6,    58,    77,     9,    77,    77,    75,    59,    77,
      61,    62,   357,    47,   188,    59,   190,    59,    90,   191,
     194,    77,    75,   195,   198,     6,     7,   201,    77,    77,
      59,   205,   206,   207,     6,    59,   221,   211,   212,   213,
     225,    77,   227,    77,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    49,   545,     3,     3,   110,
     111,     3,     3,   552,     3,     3,     6,   239,   240,    77,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   239,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,     3,    47,   454,
     282,   283,     3,     3,     3,   287,   286,    75,     4,     5,
       6,     7,     8,     3,     1,    59,     3,    61,    62,   301,
     300,    77,    77,     3,     3,    77,    77,   309,    77,    77,
      26,    27,     1,     3,     6,     4,     5,   638,     7,     8,
       1,    37,     3,     3,     3,     1,    33,     3,     3,   666,
       3,   309,    48,     1,    50,     3,    52,   339,     3,    55,
      56,     3,     3,    59,    60,    61,   110,   111,    47,     3,
       1,    77,    59,     3,     1,     6,    77,    77,    77,     6,
     364,     4,     3,     6,   368,    81,    55,    56,   370,   371,
       3,    47,    47,     6,   376,   390,     9,    93,    94,    47,
       3,     3,    98,     1,     3,    47,    47,     3,     6,     7,
     106,     3,    75,    47,   110,   111,   112,    47,   719,   391,
     404,   403,   406,   407,   408,   409,   410,   411,   412,   413,
     414,   415,   416,   417,     3,   419,   420,   421,   422,   423,
     424,   425,   426,   427,   755,   429,   430,   431,   432,   433,
     434,   435,   436,   437,   438,   439,   440,   441,   442,   443,
     444,   445,   763,   447,     1,    59,   450,    61,    62,     6,
       3,    75,     4,     5,     6,     7,     8,     3,     3,     1,
       1,     3,     3,   465,    75,   469,    77,     3,     9,   484,
     485,     1,    14,     3,     1,    27,     3,     4,     5,     6,
       7,     8,    23,    24,     3,   489,   490,     1,   492,   103,
     104,   105,     6,     1,   496,     3,   110,   111,     3,    26,
      27,     3,   517,    55,    56,    59,    47,    61,    62,     3,
      37,     4,     5,     1,     7,     8,     3,    47,     6,    33,
       1,    48,     3,    50,     4,    52,     6,   531,    55,    56,
       3,   533,    59,   535,    61,   537,   538,    60,     4,    47,
       6,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   556,     3,    81,   533,   110,   111,     1,   537,
       3,     3,    55,    56,   579,   580,    93,    94,   570,     3,
     572,    98,     1,   577,     3,     3,     1,     6,     1,   106,
       3,     6,     1,   110,   111,   112,     1,     6,     3,     4,
       5,     6,     7,     8,     1,     1,     1,    59,     6,    61,
      62,     6,     3,   603,    47,     1,     3,     3,     1,     3,
       3,    26,    27,     9,     6,   619,     6,   621,    59,     3,
      61,    62,    37,    77,    47,    63,     6,    23,    24,     3,
     622,   623,     3,    48,     3,    50,     6,    52,     6,     6,
      55,    56,   104,   105,    59,     6,    61,    60,   110,   111,
       3,    47,   656,   657,    47,    75,   671,    98,    99,   100,
     101,   102,   103,   104,   105,     3,    81,     3,    33,   110,
     111,     3,     6,     3,     3,     3,     3,     3,    93,    94,
     672,     6,     4,    98,   684,     9,     3,    59,     9,    61,
      62,   106,     3,     9,     9,   110,   111,   112,     3,    30,
     702,    58,     3,    59,   696,     3,     6,     3,     3,     3,
     725,    77,     3,   728,    76,     3,     3,     3,   733,     3,
     735,     3,    77,    58,    47,    97,    98,    99,   100,   101,
     102,   103,   104,   105,   726,    75,     3,   729,   110,   111,
      58,     6,   734,     6,   736,     9,     3,     9,     3,     3,
     754,     3,   744,     9,   756,   747,    58,     3,   750,     6,
       3,   776,    75,     3,   779,     3,     3,   782,     3,     0,
       1,   786,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,   787,    15,     3,     3,    18,     3,     3,
       6,    22,    23,     3,    25,    26,    27,    28,    29,     3,
      31,    32,   663,    34,    35,    36,    37,   696,    39,    40,
      41,    42,    43,    44,    45,    46,   384,    48,   669,    50,
      51,    52,    53,    54,    55,    56,    57,   515,    59,   678,
      61,   683,   464,   349,   354,   456,   640,   763,   552,   552,
     552,   765,   462,    32,   186,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      19,    20,    21,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    57,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      47,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,     4,
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
      -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      23,    24,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    47,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    23,    24,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    47,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    16,    17,    18,
      -1,    -1,    59,    22,    61,    62,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    57,    -1,
      59,    -1,    61,   100,   101,   102,   103,   104,   105,    -1,
      -1,    -1,    -1,   110,   111,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    30,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,     4,
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
      -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    57,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,     4,
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
      -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,
       1,    -1,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,     1,    -1,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    -1,    15,    -1,    -1,    18,
      -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,
      29,    -1,    31,    32,    -1,    34,    35,    36,    37,    -1,
      -1,    40,    -1,    -1,    43,    -1,    -1,    -1,    -1,    48,
      -1,    50,    51,    52,    53,    54,    55,    56,    57,    -1,
      59,    -1,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    93,    94,    -1,    -1,    -1,    98,
      -1,    -1,    -1,    -1,    -1,   104,    -1,   106,    -1,    -1,
      -1,   110,   111,   112,     1,    -1,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    15,    -1,
      -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,
      27,    28,    29,    -1,    31,    32,    -1,    34,    35,    36,
      37,    -1,    -1,    40,    -1,    -1,    43,    -1,    -1,    -1,
      -1,    48,    -1,    50,    51,    52,    53,    54,    55,    56,
      57,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    -1,    -1,   104,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,     1,    -1,     3,     4,
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
      -1,   106,    -1,    -1,    -1,   110,   111,   112,     1,    -1,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,
      -1,    -1,    25,    26,    27,    28,    29,    -1,    31,    32,
      -1,    34,    35,    36,    37,    -1,    -1,    40,    -1,    -1,
      43,    -1,    -1,    -1,    -1,    48,    -1,    50,    51,    52,
      53,    54,    55,    56,    57,    -1,    59,    -1,    61,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,    -1,
      -1,   104,    -1,   106,    -1,    -1,    -1,   110,   111,   112,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    10,
      11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,
      -1,    22,    -1,    -1,    25,    26,    27,    28,    29,    -1,
      31,    32,    -1,    34,    35,    36,    37,    -1,    -1,    40,
      -1,    -1,    43,    -1,    -1,    -1,    -1,    48,    -1,    50,
      51,    52,    53,    54,    55,    56,    57,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,   104,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    60,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      26,    27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,
      56,    -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,
      -1,    -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,
     106,    -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,
      -1,    52,    -1,    -1,    55,    56,     1,    -1,    59,    -1,
      61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,
      -1,    -1,    47,    -1,    -1,   106,    -1,    -1,    -1,   110,
     111,   112,    -1,    -1,    59,    -1,    61,    62,     3,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    47,    -1,    -1,   110,   111,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    59,    -1,    61,    62,     3,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    47,    -1,    -1,   110,   111,    -1,    -1,    -1,
      -1,     3,    -1,    -1,    59,    -1,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,   110,   111,    59,    -1,    61,
      62,     3,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    77,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    47,    -1,    -1,   110,   111,
      -1,    -1,    -1,    -1,     3,    -1,    -1,    59,    -1,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,     3,   110,   111,
      59,    -1,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    -1,
       3,   110,   111,    59,    -1,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    -1,     3,   110,   111,    59,    -1,    61,    62,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,     3,   110,   111,    59,
      -1,    61,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    -1,     3,
     110,   111,    59,    -1,    61,    62,    -1,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,    -1,     3,   110,   111,    59,    -1,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,     3,   110,   111,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    -1,    -1,    -1,     3,   110,
     111,    59,    -1,    61,    62,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    -1,    -1,
      -1,     3,   110,   111,    59,    -1,    61,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    -1,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,    -1,    -1,    -1,    -1,   110,   111,    59,    -1,    61,
      62,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      92,    -1,    -1,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,   105,    -1,    -1,    -1,    -1,   110,   111,
       4,     5,     6,     7,     8,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    26,    27,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    47,    48,    -1,    50,    -1,    52,    26,
      27,    55,    56,    -1,    -1,    59,    60,    61,    -1,    63,
      37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      47,    48,    -1,    50,    -1,    52,    -1,    81,    55,    56,
      -1,    -1,    59,    -1,    61,    -1,    -1,    -1,    -1,    93,
      94,    -1,    -1,    -1,    98,    -1,    -1,     4,     5,     6,
       7,     8,   106,    -1,    81,    -1,   110,   111,   112,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    93,    94,    -1,    26,
      27,    98,    -1,    -1,    -1,   102,    -1,    -1,    -1,   106,
      37,    -1,    -1,   110,   111,   112,    -1,    -1,    -1,    -1,
      -1,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    60,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    58,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    -1,
      37,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,    48,    -1,    50,    -1,    52,    -1,    -1,    55,    56,
      -1,    -1,    59,    60,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    26,    27,    93,    94,    -1,    -1,
      -1,    98,    -1,    -1,    -1,    37,    -1,    -1,    -1,   106,
      -1,    -1,    -1,   110,   111,   112,    48,    -1,    50,    -1,
      52,    -1,    -1,    55,    56,    -1,    -1,    59,    -1,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    93,    94,    -1,    -1,    -1,    98,    -1,    -1,    47,
      -1,    -1,    -1,    -1,   106,    -1,    -1,    -1,   110,   111,
     112,    59,    -1,    61,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    -1,    -1,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,   105,    47,    -1,
      -1,    -1,   110,   111,    -1,    -1,    -1,    -1,    -1,    -1,
      59,    60,    61,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    47,    -1,    -1,
      -1,   110,   111,    -1,    -1,    -1,    -1,    -1,    -1,    59,
      60,    61,    62,    -1,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    -1,    -1,
     110,   111,    58,    59,    -1,    61,    62,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      -1,    -1,    -1,    -1,   110,   111,    59,    60,    61,    62,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    -1,    -1,    -1,    -1,   110,   111,    59,
      -1,    61,    62,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    -1,    -1,    -1,    -1,
     110,   111,    59,    60,    61,    62,    -1,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    -1,
      -1,    -1,    -1,   110,   111,    59,    60,    61,    62,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    -1,    -1,    -1,    -1,   110,   111,    59,    -1,
      61,    62,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    -1,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,   105,    59,    -1,    61,    62,   110,
     111,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    75,    -1,    77,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      -1,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,    59,    -1,    61,    62,   110,   111,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    -1,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,    59,
      -1,    61,    62,   110,   111,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    92,    -1,    -1,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,   105,    59,    -1,    61,    62,
     110,   111,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    -1,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,    59,    -1,    61,    62,   110,   111,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    92,    -1,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,   105,
      59,    -1,    61,    62,   110,   111,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    88,
      89,    90,    91,    92,    -1,    -1,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,    -1,    -1,    -1,
      -1,   110,   111
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   114,   115,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    29,    31,    32,    34,    35,    36,
      37,    39,    40,    41,    42,    43,    44,    45,    46,    48,
      50,    51,    52,    53,    54,    55,    56,    57,    59,    61,
      81,    85,    93,    94,    98,   104,   106,   110,   111,   112,
     116,   117,   119,   120,   121,   123,   124,   126,   127,   128,
     130,   131,   139,   140,   141,   146,   147,   148,   155,   157,
     170,   172,   181,   182,   184,   191,   192,   193,   195,   197,
     205,   206,   207,   208,   210,   213,   216,   217,   218,   221,
     239,   244,   249,   253,   254,   255,   256,   257,   258,   259,
     261,   264,   266,   269,   270,   271,   272,   273,     3,     3,
       1,   122,   255,     1,    48,   257,     1,     3,     1,     3,
      14,     1,   257,     1,   255,   275,     1,   257,     1,     1,
     257,     1,   257,   273,     1,     3,    47,     1,   257,     1,
       6,     1,     6,     1,     3,   257,   250,   267,     1,     6,
       7,     1,   257,   259,     1,     6,     1,     3,    47,     1,
     257,     1,     3,     6,   209,     1,     6,    33,   212,     1,
       6,   214,   215,     1,     6,   262,   265,     1,   257,    60,
     257,   274,    47,     6,   257,    47,    60,    63,   257,   273,
     276,   257,     1,     3,   273,   257,   257,   257,     1,     3,
     273,   257,   257,   257,     4,   255,   125,    32,    34,    48,
     120,   129,   120,   156,   171,   183,    49,   200,   203,   204,
     120,     1,    59,     1,     6,   219,   220,   219,     3,    59,
      61,    62,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    78,    79,    80,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,   105,   110,
     111,   258,    75,    77,     3,     3,    77,    75,     3,    47,
       3,    47,     3,     3,     3,     3,    47,     3,    47,     3,
      77,    90,     3,     3,     3,     3,     3,     3,     1,    76,
       3,   120,     3,     3,     3,   222,     3,   245,     3,     3,
       6,   251,   252,     1,     6,   198,   199,   268,     3,     3,
       3,     3,     3,     3,    75,     3,    47,     3,     3,    90,
       3,     3,    77,     3,     6,     7,   211,     1,     3,    33,
      77,     3,    75,     3,    77,     3,     1,    59,   263,   263,
       3,     3,     1,    60,   257,   240,    58,    60,   257,    60,
      47,    63,     1,    60,     1,    60,    77,     3,     3,     3,
       3,   138,   138,   158,   173,   138,     1,     3,    47,   138,
     201,   202,     3,     1,   198,     3,     3,    77,     9,   219,
       3,    58,   273,   102,   257,     6,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   257,   257,     1,   257,
     257,   257,   257,   257,   257,   257,   257,   257,     6,   257,
     257,   257,   257,   257,   257,   257,   257,   257,   257,   257,
     257,   257,   257,   257,   257,   257,   255,   257,   255,     1,
     257,     3,   273,     1,    59,   223,   224,     1,    33,   226,
     246,     3,    77,     3,    77,    63,     1,   254,     1,   257,
       6,     3,    75,     3,   211,     6,     4,     6,     7,    98,
     118,   215,     3,   198,   200,   200,     3,    60,    60,   257,
     257,    60,   257,     9,   120,    16,    17,   132,   134,   135,
     137,     1,     3,    23,    24,   159,   164,   166,     1,     3,
      23,   164,   174,    30,   185,   186,   187,   188,     3,    47,
       9,   138,   120,   196,     1,    58,     6,     3,     3,     1,
      58,   257,    60,    77,     1,    47,     3,    77,    75,     3,
       3,    47,     3,     3,   198,   232,   226,     3,     6,   227,
     228,     3,   247,     1,   252,   199,   257,     3,     3,     3,
       3,     6,     3,    75,     4,     1,    58,   138,   138,   241,
      47,    60,    63,     3,     1,     3,     1,   257,     9,   133,
     136,     3,     1,     6,     7,     8,   118,   168,   169,     1,
       9,   165,     3,     1,     4,     6,   179,   180,     9,     1,
       3,     4,     6,    90,   189,   190,     9,   187,   138,     3,
       9,    58,   194,     3,    47,   260,    60,   273,     1,   257,
     273,   257,   142,   143,     1,    58,     3,     6,    38,    48,
      49,    92,   192,   233,   234,   236,   237,     3,    59,   229,
      77,     3,   192,   234,   236,   237,   248,     3,     6,     3,
       9,     9,     3,     6,     9,   242,   257,   257,     3,     3,
       3,     3,   138,   138,     3,    47,    76,     3,    47,    77,
       3,     3,    47,   167,     3,    47,     3,    47,    77,     3,
       3,   255,     3,    77,    90,     3,     3,    47,    58,    58,
       3,    19,    20,    21,   120,   144,   145,   149,   151,   153,
     120,   225,    75,     3,     6,     1,     6,    81,   238,     9,
       6,    27,   230,   231,   254,   228,     9,     3,     3,    75,
      77,   243,     3,    60,   132,   162,   163,   118,   160,   161,
     169,   138,   120,   177,   178,   175,   176,   180,     3,   190,
     255,     3,     1,     3,    47,     1,     3,    47,     1,     3,
      47,     9,   144,    58,   257,   235,    75,     3,     6,     3,
      77,     3,    58,    77,     3,   254,   138,   120,   138,   120,
     138,   120,   138,   120,     3,     3,   150,   120,     3,   152,
     120,     3,   154,   120,     3,     3,   200,   257,     6,    81,
     231,   243,   138,   138,   138,   138,     3,     6,     9,     9,
       9,     9,     3,     3,     3,     3
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
#line 209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].stringp) != 0 )
            COMPILER->addLoad( *(yyvsp[(1) - (1)].stringp) );
      }
    break;

  case 10:
#line 221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:
#line 248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:
#line 253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      }
    break;

  case 21:
#line 259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         (yyval.stringp) = (yyvsp[(2) - (3)].stringp);
      }
    break;

  case 22:
#line 265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
         (yyval.stringp) = 0;
      }
    break;

  case 23:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 25:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 275 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 276 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 29:
#line 282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); }
    break;

  case 30:
#line 284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 31:
#line 290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 51:
#line 326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:
#line 332 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:
#line 341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:
#line 343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:
#line 347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:
#line 354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 60:
#line 371 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 61:
#line 375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 62:
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 63:
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 64:
#line 381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 65:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 66:
#line 396 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 67:
#line 406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 68:
#line 407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 69:
#line 411 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 70:
#line 412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 73:
#line 419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 76:
#line 429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 77:
#line 434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 79:
#line 446 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 80:
#line 447 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 82:
#line 452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 83:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 84:
#line 468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 85:
#line 476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 86:
#line 486 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 87:
#line 495 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 88:
#line 504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 89:
#line 520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 90:
#line 528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 91:
#line 544 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(7) - (7)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(7) - (7)].fal_stat) );
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 92:
#line 554 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 93:
#line 559 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:
#line 571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 100:
#line 585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 101:
#line 598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 102:
#line 606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 103:
#line 610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 104:
#line 616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 105:
#line 622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 106:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 107:
#line 634 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 108:
#line 643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 109:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         COMPILER->pushContextSet( &f->firstBlock() );
		 // Push anyhow an empty item, that is needed for to check again for thio blosk
		 f->firstBlock().push_back( new Falcon::StmtNone( LINE ) );
      }
    break;

  case 110:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 111:
#line 666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->firstBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forfirst );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->firstBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 112:
#line 675 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 113:
#line 679 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
		 // Push anyhow an empty item, that is needed for empty last blocks
		 f->lastBlock().push_back( new Falcon::StmtNone( LINE ) );
         COMPILER->pushContextSet( &f->lastBlock() );
      }
    break;

  case 114:
#line 691 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 115:
#line 692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->lastBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_forlast );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->lastBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 116:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 117:
#line 705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 118:
#line 719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 119:
#line 721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         if( ! f->middleBlock().empty() )
         {
            COMPILER->raiseError( Falcon::e_already_formiddle );
         }
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            f->middleBlock().push_back( (yyvsp[(3) - (3)].fal_stat) );
      }
    break;

  case 120:
#line 730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 121:
#line 734 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 122:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 123:
#line 751 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 124:
#line 753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 127:
#line 762 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 129:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 131:
#line 778 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 132:
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 133:
#line 790 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 135:
#line 802 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 136:
#line 812 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 138:
#line 821 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();
         if ( ! stmt->defaultBlock().empty() )
         {
            COMPILER->raiseError(Falcon::e_switch_default, "", CURRENT_LINE );
         }
         COMPILER->pushContextSet( &stmt->defaultBlock() );
      }
    break;

  case 142:
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 144:
#line 839 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 147:
#line 851 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 148:
#line 860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 149:
#line 872 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].stringp) );
         if ( ! stmt->addStringCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 150:
#line 883 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (yyvsp[(1) - (3)].integer) ), new Falcon::Value( (yyvsp[(3) - (3)].integer) ) ) );
         if ( ! stmt->addRangeCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 151:
#line 894 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 152:
#line 914 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 153:
#line 922 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 154:
#line 931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 155:
#line 933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 158:
#line 942 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 160:
#line 948 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 162:
#line 958 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 163:
#line 967 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 164:
#line 971 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 166:
#line 983 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 167:
#line 993 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 171:
#line 1007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );
         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_switch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 172:
#line 1019 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 173:
#line 1040 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, (yyvsp[(4) - (5)].fal_adecl), (yyvsp[(2) - (5)].fal_adecl) );
      }
    break;

  case 174:
#line 1044 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtGive( LINE, 0, (yyvsp[(2) - (4)].fal_adecl) );
         COMPILER->raiseError(Falcon::e_syn_give );
      }
    break;

  case 175:
#line 1048 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_give ); (yyval.fal_stat) = 0; }
    break;

  case 176:
#line 1056 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 177:
#line 1063 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 178:
#line 1073 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 180:
#line 1082 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 186:
#line 1102 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 187:
#line 1120 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 188:
#line 1140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 189:
#line 1150 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 190:
#line 1161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 193:
#line 1174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *stmt = static_cast<Falcon::StmtTry *>(COMPILER->getContext());
         Falcon::Value *val = new Falcon::Value( (yyvsp[(1) - (1)].integer) );

         if ( ! stmt->addIntCase( val ) )
         {
            COMPILER->raiseError(Falcon::e_catch_clash, "", CURRENT_LINE );
            delete val;
         }
      }
    break;

  case 194:
#line 1186 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 195:
#line 1208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 196:
#line 1209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 197:
#line 1221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 198:
#line 1227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 200:
#line 1236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 201:
#line 1237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 202:
#line 1240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 204:
#line 1245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 205:
#line 1246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 206:
#line 1253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 210:
#line 1314 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 212:
#line 1331 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 213:
#line 1337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 214:
#line 1342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 215:
#line 1348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 217:
#line 1357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 219:
#line 1362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 220:
#line 1372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 221:
#line 1375 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 222:
#line 1384 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getFunction() == 0 )
            COMPILER->raiseError(Falcon::e_pass_outside );
         else
            (yyval.fal_stat) = new Falcon::StmtPass( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 223:
#line 1391 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 224:
#line 1406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(2) - (5)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_pass_in );
         (yyval.fal_stat) = 0;
      }
    break;

  case 225:
#line 1412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_pass );
         (yyval.fal_stat) = 0;
      }
    break;

  case 226:
#line 1424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // TODO: evalute const expressions on the fly.
         Falcon::Value *val = (yyvsp[(4) - (5)].fal_val); //COMPILER->exprSimplify( $4 );
         // will raise an error in case the expression is not atomic.
         COMPILER->addConstant( *(yyvsp[(2) - (5)].stringp), val, LINE );
         // we don't need the expression anymore
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 227:
#line 1434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 228:
#line 1439 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 229:
#line 1451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 230:
#line 1460 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:
#line 1467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:
#line 1475 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 233:
#line 1480 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 234:
#line 1488 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:
#line 1493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:
#line 1498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:
#line 1503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (4)].fal_genericList)->begin();
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            delete symName;
            li = li->next();
         }
         delete (yyvsp[(2) - (4)].fal_genericList);

         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 238:
#line 1517 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:
#line 1522 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:
#line 1527 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:
#line 1540 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 244:
#line 1546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 245:
#line 1558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 246:
#line 1563 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 249:
#line 1576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 250:
#line 1580 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 251:
#line 1584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 252:
#line 1598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 253:
#line 1605 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no other action:
         (yyval.fal_stat) = 0;
      }
    break;

  case 255:
#line 1613 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes ); }
    break;

  case 257:
#line 1617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_attributes, "", CURRENT_LINE ); }
    break;

  case 259:
#line 1623 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(1) - (1)].stringp) );
         }
    break;

  case 260:
#line 1627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addAttribute( (yyvsp[(3) - (3)].stringp) );
         }
    break;

  case 263:
#line 1636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_inv_attrib );
   }
    break;

  case 264:
#line 1647 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
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
      }
    break;

  case 265:
#line 1681 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 267:
#line 1709 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 270:
#line 1717 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 271:
#line 1718 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 276:
#line 1735 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 277:
#line 1768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = 0; }
    break;

  case 278:
#line 1773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
   }
    break;

  case 279:
#line 1779 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 280:
#line 1780 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 282:
#line 1786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // the symbol must be a parameter, or we raise an error
         Falcon::Symbol *sym = COMPILER->searchLocalSymbol( (yyvsp[(1) - (1)].stringp) );
         if ( sym == 0 || sym->type() != Falcon::Symbol::tparam ) {
            sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         }
         (yyval.fal_val) = new Falcon::Value( sym );
      }
    break;

  case 283:
#line 1794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 287:
#line 1804 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 288:
#line 1807 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
   }
    break;

  case 290:
#line 1829 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 291:
#line 1853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());

         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 292:
#line 1864 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
   }
    break;

  case 293:
#line 1886 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
   }
    break;

  case 296:
#line 1916 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_hasdef );
   }
    break;

  case 297:
#line 1923 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();

         // The symbolmay be undefined or defined; it's not our task to define it here.
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) ) );
      }
    break;

  case 298:
#line 1931 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(2) - (2)].stringp) ) );
      }
    break;

  case 299:
#line 1937 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->has().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) ) );
      }
    break;

  case 300:
#line 1943 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         clsdef->hasnt().pushBack( COMPILER->addGlobalSymbol( (yyvsp[(4) - (4)].stringp) ) );
      }
    break;

  case 301:
#line 1956 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
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
      }
    break;

  case 302:
#line 1990 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 306:
#line 2007 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 307:
#line 2012 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 310:
#line 2027 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ClassDef *def = new Falcon::ClassDef;
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
      }
    break;

  case 311:
#line 2067 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 313:
#line 2092 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 317:
#line 2104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 318:
#line 2107 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
   }
    break;

  case 320:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 321:
#line 2140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // raise an error if we are not in a local context
         if ( ! COMPILER->isLocalContext() )
         {
            COMPILER->raiseError(Falcon::e_global_notin_func, "", LINE );
         }
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
      }
    break;

  case 324:
#line 2155 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 325:
#line 2162 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 326:
#line 2177 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 327:
#line 2178 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 328:
#line 2179 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 329:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 330:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 331:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 332:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 333:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 334:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 335:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
     }
    break;

  case 337:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 338:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSender(); }
    break;

  case 341:
#line 2231 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 342:
#line 2232 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 343:
#line 2233 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 344:
#line 2234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 345:
#line 2235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 346:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 347:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 348:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 349:
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 350:
#line 2240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 351:
#line 2241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 352:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 353:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 354:
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 355:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 356:
#line 2246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 357:
#line 2247 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 358:
#line 2248 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 359:
#line 2249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 360:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 361:
#line 2251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 362:
#line 2252 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 363:
#line 2253 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 364:
#line 2254 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 365:
#line 2255 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 366:
#line 2256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 367:
#line 2257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_has, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 368:
#line 2258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_hasnt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 369:
#line 2259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 370:
#line 2260 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 371:
#line 2261 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 372:
#line 2262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 373:
#line 2263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 374:
#line 2264 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 375:
#line 2265 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 382:
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 383:
#line 2278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 384:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 385:
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 386:
#line 2293 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 389:
#line 2305 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 390:
#line 2310 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 391:
#line 2317 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 392:
#line 2318 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 393:
#line 2319 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 394:
#line 2320 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 395:
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 396:
#line 2322 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 397:
#line 2323 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 398:
#line 2324 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 399:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 400:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 401:
#line 2327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 402:
#line 2328 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 403:
#line 2333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 404:
#line 2336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 405:
#line 2339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 406:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 407:
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 408:
#line 2352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 409:
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 410:
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 411:
#line 2363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 412:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 413:
#line 2406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 415:
#line 2414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 416:
#line 2418 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 417:
#line 2425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 418:
#line 2458 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 419:
#line 2470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
      }
    break;

  case 420:
#line 2502 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            COMPILER->addStatement( new Falcon::StmtReturn( LINE, (yyvsp[(5) - (5)].fal_val) ) );
            COMPILER->checkLocalUndefined();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 422:
#line 2514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_lambda );
      }
    break;

  case 423:
#line 2523 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 424:
#line 2528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 425:
#line 2535 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 426:
#line 2542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 427:
#line 2551 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 428:
#line 2553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 429:
#line 2557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 430:
#line 2564 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 431:
#line 2566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 432:
#line 2570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 433:
#line 2578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 434:
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 435:
#line 2581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 436:
#line 2588 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 437:
#line 2589 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 438:
#line 2593 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 439:
#line 2594 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (2)].fal_adecl)->pushBack( (yyvsp[(2) - (2)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (2)].fal_adecl); }
    break;

  case 440:
#line 2598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 441:
#line 2604 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 442:
#line 2611 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 443:
#line 2612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6322 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2616 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */



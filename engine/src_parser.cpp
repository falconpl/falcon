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
     SELF = 281,
     FSELF = 282,
     TRY = 283,
     CATCH = 284,
     RAISE = 285,
     CLASS = 286,
     FROM = 287,
     OBJECT = 288,
     RETURN = 289,
     GLOBAL = 290,
     LAMBDA = 291,
     INIT = 292,
     LOAD = 293,
     LAUNCH = 294,
     CONST_KW = 295,
     EXPORT = 296,
     IMPORT = 297,
     DIRECTIVE = 298,
     COLON = 299,
     FUNCDECL = 300,
     STATIC = 301,
     INNERFUNC = 302,
     FORDOT = 303,
     LISTPAR = 304,
     LOOP = 305,
     ENUM = 306,
     TRUE_TOKEN = 307,
     FALSE_TOKEN = 308,
     OUTER_STRING = 309,
     CLOSEPAR = 310,
     OPENPAR = 311,
     CLOSESQUARE = 312,
     OPENSQUARE = 313,
     DOT = 314,
     OPEN_GRAPH = 315,
     CLOSE_GRAPH = 316,
     ARROW = 317,
     VBAR = 318,
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
     OP_AS = 331,
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
     DIESIS = 347,
     ATSIGN = 348,
     CAP_CAP = 349,
     VBAR_VBAR = 350,
     AMPER_AMPER = 351,
     MINUS = 352,
     PLUS = 353,
     PERCENT = 354,
     SLASH = 355,
     STAR = 356,
     POW = 357,
     SHR = 358,
     SHL = 359,
     CAP_XOROOB = 360,
     CAP_ISOOB = 361,
     CAP_DEOOB = 362,
     CAP_OOB = 363,
     CAP_EVAL = 364,
     TILDE = 365,
     NEG = 366,
     AMPER = 367,
     DECREMENT = 368,
     INCREMENT = 369,
     DOLLAR = 370
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
#define SELF 281
#define FSELF 282
#define TRY 283
#define CATCH 284
#define RAISE 285
#define CLASS 286
#define FROM 287
#define OBJECT 288
#define RETURN 289
#define GLOBAL 290
#define LAMBDA 291
#define INIT 292
#define LOAD 293
#define LAUNCH 294
#define CONST_KW 295
#define EXPORT 296
#define IMPORT 297
#define DIRECTIVE 298
#define COLON 299
#define FUNCDECL 300
#define STATIC 301
#define INNERFUNC 302
#define FORDOT 303
#define LISTPAR 304
#define LOOP 305
#define ENUM 306
#define TRUE_TOKEN 307
#define FALSE_TOKEN 308
#define OUTER_STRING 309
#define CLOSEPAR 310
#define OPENPAR 311
#define CLOSESQUARE 312
#define OPENSQUARE 313
#define DOT 314
#define OPEN_GRAPH 315
#define CLOSE_GRAPH 316
#define ARROW 317
#define VBAR 318
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
#define OP_AS 331
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
#define DIESIS 347
#define ATSIGN 348
#define CAP_CAP 349
#define VBAR_VBAR 350
#define AMPER_AMPER 351
#define MINUS 352
#define PLUS 353
#define PERCENT 354
#define SLASH 355
#define STAR 356
#define POW 357
#define SHR 358
#define SHL 359
#define CAP_XOROOB 360
#define CAP_ISOOB 361
#define CAP_DEOOB 362
#define CAP_OOB 363
#define CAP_EVAL 364
#define TILDE 365
#define NEG 366
#define AMPER 367
#define DECREMENT 368
#define INCREMENT 369
#define DOLLAR 370




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
#line 391 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 404 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6608

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  116
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  159
/* YYNRULES -- Number of rules.  */
#define YYNRULES  438
/* YYNRULES -- Number of states.  */
#define YYNSTATES  801

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   370

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
     115
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
     127,   133,   137,   141,   142,   148,   151,   155,   159,   163,
     167,   168,   176,   180,   184,   185,   187,   188,   195,   198,
     202,   206,   210,   214,   215,   217,   218,   222,   225,   229,
     230,   235,   239,   243,   244,   247,   250,   254,   257,   261,
     265,   266,   273,   274,   281,   287,   291,   294,   299,   304,
     309,   313,   314,   317,   320,   321,   324,   326,   328,   330,
     332,   336,   340,   344,   347,   351,   354,   358,   362,   364,
     365,   372,   376,   380,   381,   388,   392,   396,   397,   404,
     408,   412,   413,   420,   424,   428,   429,   432,   436,   438,
     439,   445,   446,   452,   453,   459,   460,   466,   467,   468,
     472,   473,   475,   478,   481,   484,   486,   490,   492,   494,
     496,   500,   502,   503,   510,   514,   518,   519,   522,   526,
     528,   529,   535,   536,   542,   543,   549,   550,   556,   558,
     562,   563,   565,   567,   571,   572,   579,   582,   586,   587,
     589,   591,   594,   597,   600,   605,   609,   615,   619,   621,
     625,   627,   629,   633,   637,   643,   646,   652,   653,   661,
     665,   671,   672,   679,   682,   683,   685,   689,   691,   692,
     693,   699,   700,   704,   707,   711,   714,   718,   722,   726,
     732,   738,   742,   745,   749,   753,   755,   759,   763,   769,
     775,   783,   791,   799,   807,   812,   817,   822,   827,   834,
     841,   845,   847,   851,   855,   859,   861,   865,   869,   873,
     877,   878,   886,   890,   893,   894,   898,   899,   905,   906,
     909,   911,   915,   918,   919,   922,   926,   927,   930,   932,
     934,   936,   938,   939,   947,   953,   958,   959,   967,   968,
     971,   973,   978,   981,   983,   985,   986,   994,   997,  1000,
    1001,  1004,  1006,  1008,  1010,  1012,  1013,  1018,  1020,  1024,
    1028,  1030,  1033,  1037,  1041,  1043,  1045,  1047,  1049,  1051,
    1053,  1055,  1057,  1059,  1061,  1063,  1065,  1068,  1071,  1074,
    1077,  1081,  1085,  1089,  1093,  1097,  1101,  1105,  1109,  1113,
    1117,  1121,  1125,  1128,  1132,  1135,  1138,  1141,  1144,  1148,
    1152,  1156,  1160,  1164,  1168,  1172,  1175,  1179,  1183,  1187,
    1190,  1193,  1196,  1199,  1202,  1205,  1208,  1211,  1214,  1216,
    1218,  1220,  1222,  1224,  1226,  1229,  1231,  1236,  1242,  1246,
    1248,  1250,  1254,  1260,  1264,  1268,  1272,  1276,  1280,  1284,
    1288,  1292,  1296,  1300,  1304,  1308,  1312,  1317,  1322,  1328,
    1336,  1341,  1345,  1346,  1353,  1354,  1361,  1362,  1369,  1374,
    1378,  1381,  1384,  1387,  1390,  1391,  1398,  1404,  1410,  1415,
    1419,  1422,  1426,  1430,  1433,  1437,  1441,  1445,  1449,  1454,
    1456,  1460,  1462,  1466,  1467,  1469,  1471,  1475,  1479
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     117,     0,    -1,   118,    -1,    -1,   118,   119,    -1,   120,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   122,    -1,
     219,    -1,   200,    -1,   222,    -1,   241,    -1,   236,    -1,
     123,    -1,   214,    -1,   215,    -1,   217,    -1,     4,    -1,
      97,     4,    -1,    38,     6,     3,    -1,    38,     7,     3,
      -1,    38,     1,     3,    -1,   124,    -1,     3,    -1,    45,
       1,     3,    -1,    33,     1,     3,    -1,    31,     1,     3,
      -1,     1,     3,    -1,   254,     3,    -1,   270,    75,   254,
       3,    -1,   270,    75,   254,    78,   270,     3,    -1,   126,
      -1,   127,    -1,   131,    -1,   147,    -1,   164,    -1,   179,
      -1,   134,    -1,   145,    -1,   146,    -1,   190,    -1,   199,
      -1,   250,    -1,   246,    -1,   213,    -1,   155,    -1,   156,
      -1,   157,    -1,   252,    75,   254,    -1,   125,    78,   252,
      75,   254,    -1,    10,   125,     3,    -1,    10,     1,     3,
      -1,    -1,   129,   128,   144,     9,     3,    -1,   130,   123,
      -1,    11,   254,     3,    -1,    11,     1,     3,    -1,    11,
     254,    44,    -1,    11,     1,    44,    -1,    -1,    50,     3,
     132,   144,     9,   133,     3,    -1,    50,    44,   123,    -1,
      50,     1,     3,    -1,    -1,   254,    -1,    -1,   136,   135,
     144,   138,     9,     3,    -1,   137,   123,    -1,    15,   254,
       3,    -1,    15,     1,     3,    -1,    15,   254,    44,    -1,
      15,     1,    44,    -1,    -1,   141,    -1,    -1,   140,   139,
     144,    -1,    16,     3,    -1,    16,     1,     3,    -1,    -1,
     143,   142,   144,   138,    -1,    17,   254,     3,    -1,    17,
       1,     3,    -1,    -1,   144,   123,    -1,    12,     3,    -1,
      12,     1,     3,    -1,    13,     3,    -1,    13,    14,     3,
      -1,    13,     1,     3,    -1,    -1,    18,   273,    91,   254,
     148,   150,    -1,    -1,    18,   252,    75,   151,   149,   150,
      -1,    18,   273,    91,     1,     3,    -1,    18,     1,     3,
      -1,    44,   123,    -1,     3,   153,     9,     3,    -1,   254,
      77,   254,   152,    -1,   254,    77,   254,     1,    -1,   254,
      77,     1,    -1,    -1,    78,   254,    -1,    78,     1,    -1,
      -1,   154,   153,    -1,   123,    -1,   158,    -1,   160,    -1,
     162,    -1,    48,   254,     3,    -1,    48,     1,     3,    -1,
     103,   270,     3,    -1,   103,     3,    -1,    86,   270,     3,
      -1,    86,     3,    -1,   103,     1,     3,    -1,    86,     1,
       3,    -1,    54,    -1,    -1,    19,     3,   159,   144,     9,
       3,    -1,    19,    44,   123,    -1,    19,     1,     3,    -1,
      -1,    20,     3,   161,   144,     9,     3,    -1,    20,    44,
     123,    -1,    20,     1,     3,    -1,    -1,    21,     3,   163,
     144,     9,     3,    -1,    21,    44,   123,    -1,    21,     1,
       3,    -1,    -1,   166,   165,   167,   173,     9,     3,    -1,
      22,   254,     3,    -1,    22,     1,     3,    -1,    -1,   167,
     168,    -1,   167,     1,     3,    -1,     3,    -1,    -1,    23,
     177,     3,   169,   144,    -1,    -1,    23,   177,    44,   170,
     123,    -1,    -1,    23,     1,     3,   171,   144,    -1,    -1,
      23,     1,    44,   172,   123,    -1,    -1,    -1,   175,   174,
     176,    -1,    -1,    24,    -1,    24,     1,    -1,     3,   144,
      -1,    44,   123,    -1,   178,    -1,   177,    78,   178,    -1,
       8,    -1,   121,    -1,     7,    -1,   121,    77,   121,    -1,
       6,    -1,    -1,   181,   180,   182,   173,     9,     3,    -1,
      25,   254,     3,    -1,    25,     1,     3,    -1,    -1,   182,
     183,    -1,   182,     1,     3,    -1,     3,    -1,    -1,    23,
     188,     3,   184,   144,    -1,    -1,    23,   188,    44,   185,
     123,    -1,    -1,    23,     1,     3,   186,   144,    -1,    -1,
      23,     1,    44,   187,   123,    -1,   189,    -1,   188,    78,
     189,    -1,    -1,     4,    -1,     6,    -1,    28,    44,   123,
      -1,    -1,   192,   191,   144,   193,     9,     3,    -1,    28,
       3,    -1,    28,     1,     3,    -1,    -1,   194,    -1,   195,
      -1,   194,   195,    -1,   196,   144,    -1,    29,     3,    -1,
      29,    91,   252,     3,    -1,    29,   197,     3,    -1,    29,
     197,    91,   252,     3,    -1,    29,     1,     3,    -1,   198,
      -1,   197,    78,   198,    -1,     4,    -1,     6,    -1,    30,
     254,     3,    -1,    30,     1,     3,    -1,   201,   208,   144,
       9,     3,    -1,   203,   123,    -1,   205,    56,   206,    55,
       3,    -1,    -1,   205,    56,   206,     1,   202,    55,     3,
      -1,   205,     1,     3,    -1,   205,    56,   206,    55,    44,
      -1,    -1,   205,    56,     1,   204,    55,    44,    -1,    45,
       6,    -1,    -1,   207,    -1,   206,    78,   207,    -1,     6,
      -1,    -1,    -1,   211,   209,   144,     9,     3,    -1,    -1,
     212,   210,   123,    -1,    46,     3,    -1,    46,     1,     3,
      -1,    46,    44,    -1,    46,     1,    44,    -1,    39,   256,
       3,    -1,    39,     1,     3,    -1,    40,     6,    75,   251,
       3,    -1,    40,     6,    75,     1,     3,    -1,    40,     1,
       3,    -1,    41,     3,    -1,    41,   216,     3,    -1,    41,
       1,     3,    -1,     6,    -1,   216,    78,     6,    -1,    42,
     218,     3,    -1,    42,   218,    32,     6,     3,    -1,    42,
     218,    32,     7,     3,    -1,    42,   218,    32,     6,    76,
       6,     3,    -1,    42,   218,    32,     7,    76,     6,     3,
      -1,    42,   218,    32,     6,    91,     6,     3,    -1,    42,
     218,    32,     7,    91,     6,     3,    -1,    42,     6,     1,
       3,    -1,    42,   218,     1,     3,    -1,    42,    32,     6,
       3,    -1,    42,    32,     7,     3,    -1,    42,    32,     6,
      76,     6,     3,    -1,    42,    32,     7,    76,     6,     3,
      -1,    42,     1,     3,    -1,     6,    -1,   218,    78,     6,
      -1,    43,   220,     3,    -1,    43,     1,     3,    -1,   221,
      -1,   220,    78,   221,    -1,     6,    75,     6,    -1,     6,
      75,     7,    -1,     6,    75,   121,    -1,    -1,    31,     6,
     223,   224,   231,     9,     3,    -1,   225,   227,     3,    -1,
       1,     3,    -1,    -1,    56,   206,    55,    -1,    -1,    56,
     206,     1,   226,    55,    -1,    -1,    32,   228,    -1,   229,
      -1,   228,    78,   229,    -1,     6,   230,    -1,    -1,    56,
      55,    -1,    56,   270,    55,    -1,    -1,   231,   232,    -1,
       3,    -1,   200,    -1,   235,    -1,   233,    -1,    -1,    37,
       3,   234,   208,   144,     9,     3,    -1,    46,     6,    75,
     254,     3,    -1,     6,    75,   254,     3,    -1,    -1,    51,
       6,   237,     3,   238,     9,     3,    -1,    -1,   238,   239,
      -1,     3,    -1,     6,    75,   251,   240,    -1,     6,   240,
      -1,     3,    -1,    78,    -1,    -1,    33,     6,   242,   243,
     244,     9,     3,    -1,   227,     3,    -1,     1,     3,    -1,
      -1,   244,   245,    -1,     3,    -1,   200,    -1,   235,    -1,
     233,    -1,    -1,    35,   247,   248,     3,    -1,   249,    -1,
     248,    78,   249,    -1,   248,    78,     1,    -1,     6,    -1,
      34,     3,    -1,    34,   254,     3,    -1,    34,     1,     3,
      -1,     8,    -1,    52,    -1,    53,    -1,     4,    -1,     5,
      -1,     7,    -1,     6,    -1,   252,    -1,    26,    -1,    27,
      -1,   251,    -1,   253,    -1,   112,     6,    -1,   112,     4,
      -1,   112,    26,    -1,    97,   254,    -1,     6,    63,   254,
      -1,   254,    98,   254,    -1,   254,    97,   254,    -1,   254,
     101,   254,    -1,   254,   100,   254,    -1,   254,    99,   254,
      -1,   254,   102,   254,    -1,   254,    96,   254,    -1,   254,
      95,   254,    -1,   254,    94,   254,    -1,   254,   104,   254,
      -1,   254,   103,   254,    -1,   110,   254,    -1,   254,    87,
     254,    -1,   254,   114,    -1,   114,   254,    -1,   254,   113,
      -1,   113,   254,    -1,   254,    88,   254,    -1,   254,    86,
     254,    -1,   254,    85,   254,    -1,   254,    84,   254,    -1,
     254,    83,   254,    -1,   254,    81,   254,    -1,   254,    80,
     254,    -1,    82,   254,    -1,   254,    91,   254,    -1,   254,
      90,   254,    -1,   254,    89,     6,    -1,   115,   252,    -1,
     115,     4,    -1,    93,   254,    -1,    92,   254,    -1,   109,
     254,    -1,   108,   254,    -1,   107,   254,    -1,   106,   254,
      -1,   105,   254,    -1,   258,    -1,   260,    -1,   264,    -1,
     256,    -1,   266,    -1,   268,    -1,   254,   255,    -1,   267,
      -1,   254,    58,   254,    57,    -1,   254,    58,   101,   254,
      57,    -1,   254,    59,     6,    -1,   269,    -1,   255,    -1,
     254,    75,   254,    -1,   254,    75,   254,    78,   270,    -1,
     254,    74,   254,    -1,   254,    73,   254,    -1,   254,    72,
     254,    -1,   254,    71,   254,    -1,   254,    70,   254,    -1,
     254,    64,   254,    -1,   254,    69,   254,    -1,   254,    68,
     254,    -1,   254,    67,   254,    -1,   254,    65,   254,    -1,
     254,    66,   254,    -1,    56,   254,    55,    -1,    58,    44,
      57,    -1,    58,   254,    44,    57,    -1,    58,    44,   254,
      57,    -1,    58,   254,    44,   254,    57,    -1,    58,   254,
      44,   254,    44,   254,    57,    -1,   254,    56,   270,    55,
      -1,   254,    56,    55,    -1,    -1,   254,    56,   270,     1,
     257,    55,    -1,    -1,    45,   259,   262,   208,   144,     9,
      -1,    -1,    60,   261,   263,   208,   144,    61,    -1,    56,
     206,    55,     3,    -1,    56,   206,     1,    -1,     1,     3,
      -1,   206,    62,    -1,   206,     1,    -1,     1,    62,    -1,
      -1,    47,   265,   262,   208,   144,     9,    -1,   254,    79,
     254,    44,   254,    -1,   254,    79,   254,    44,     1,    -1,
     254,    79,   254,     1,    -1,   254,    79,     1,    -1,    58,
      57,    -1,    58,   270,    57,    -1,    58,   270,     1,    -1,
      49,    57,    -1,    49,   271,    57,    -1,    49,   271,     1,
      -1,    58,    62,    57,    -1,    58,   274,    57,    -1,    58,
     274,     1,    57,    -1,   254,    -1,   270,    78,   254,    -1,
     254,    -1,   271,   272,   254,    -1,    -1,    78,    -1,   252,
      -1,   273,    78,   252,    -1,   254,    62,   254,    -1,   274,
      78,   254,    62,   254,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   200,   202,   206,   207,   208,   212,   213,
     214,   219,   224,   229,   234,   239,   240,   241,   245,   246,
     250,   256,   262,   269,   270,   271,   272,   273,   274,   279,
     281,   287,   301,   302,   303,   304,   305,   306,   307,   308,
     309,   310,   311,   312,   313,   314,   315,   316,   317,   321,
     327,   335,   337,   342,   342,   356,   364,   365,   369,   370,
     374,   374,   389,   395,   401,   402,   406,   406,   421,   431,
     432,   436,   437,   441,   443,   444,   444,   453,   454,   459,
     459,   471,   472,   475,   477,   483,   492,   500,   510,   519,
     529,   528,   555,   554,   575,   580,   588,   594,   601,   607,
     611,   618,   619,   620,   623,   625,   629,   636,   637,   638,
     642,   655,   663,   667,   673,   679,   686,   691,   700,   710,
     710,   724,   733,   737,   737,   750,   759,   763,   763,   779,
     788,   792,   792,   809,   810,   817,   819,   820,   824,   826,
     825,   836,   836,   848,   848,   860,   860,   876,   879,   878,
     891,   892,   893,   896,   897,   903,   904,   908,   917,   929,
     940,   951,   972,   972,   989,   990,   997,   999,  1000,  1004,
    1006,  1005,  1016,  1016,  1029,  1029,  1041,  1041,  1059,  1060,
    1063,  1064,  1076,  1097,  1104,  1103,  1122,  1123,  1126,  1128,
    1132,  1133,  1137,  1142,  1160,  1180,  1190,  1201,  1209,  1210,
    1214,  1226,  1249,  1250,  1257,  1267,  1276,  1277,  1277,  1281,
    1285,  1286,  1286,  1293,  1347,  1349,  1350,  1354,  1369,  1372,
    1371,  1383,  1382,  1397,  1398,  1402,  1403,  1412,  1416,  1424,
    1434,  1439,  1451,  1460,  1467,  1475,  1480,  1488,  1493,  1498,
    1503,  1523,  1542,  1547,  1552,  1557,  1571,  1576,  1581,  1586,
    1591,  1599,  1605,  1617,  1622,  1630,  1631,  1635,  1639,  1643,
    1657,  1656,  1717,  1720,  1726,  1728,  1729,  1729,  1735,  1737,
    1741,  1742,  1746,  1770,  1771,  1772,  1779,  1781,  1785,  1786,
    1789,  1807,  1811,  1811,  1843,  1865,  1899,  1898,  1942,  1944,
    1948,  1949,  1954,  1961,  1961,  1970,  1969,  2036,  2037,  2043,
    2045,  2049,  2050,  2053,  2072,  2081,  2080,  2098,  2099,  2100,
    2107,  2123,  2124,  2125,  2135,  2136,  2137,  2138,  2139,  2140,
    2144,  2162,  2163,  2164,  2185,  2186,  2187,  2188,  2189,  2190,
    2191,  2192,  2193,  2194,  2195,  2196,  2197,  2198,  2199,  2200,
    2201,  2202,  2203,  2204,  2205,  2206,  2207,  2208,  2209,  2210,
    2211,  2212,  2213,  2214,  2215,  2216,  2217,  2218,  2219,  2220,
    2221,  2222,  2223,  2224,  2225,  2226,  2227,  2228,  2229,  2230,
    2231,  2232,  2233,  2234,  2236,  2241,  2245,  2250,  2256,  2265,
    2266,  2268,  2273,  2280,  2281,  2282,  2283,  2284,  2285,  2286,
    2287,  2288,  2289,  2290,  2291,  2296,  2299,  2302,  2305,  2308,
    2314,  2320,  2325,  2325,  2335,  2334,  2376,  2375,  2427,  2428,
    2432,  2439,  2440,  2444,  2452,  2451,  2499,  2504,  2511,  2518,
    2528,  2529,  2533,  2541,  2542,  2546,  2555,  2556,  2557,  2565,
    2566,  2570,  2571,  2574,  2575,  2578,  2584,  2591,  2592
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
  "SWITCH", "CASE", "DEFAULT", "SELECT", "SELF", "FSELF", "TRY", "CATCH",
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "LAMBDA", "INIT",
  "LOAD", "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE", "COLON",
  "FUNCDECL", "STATIC", "INNERFUNC", "FORDOT", "LISTPAR", "LOOP", "ENUM",
  "TRUE_TOKEN", "FALSE_TOKEN", "OUTER_STRING", "CLOSEPAR", "OPENPAR",
  "CLOSESQUARE", "OPENSQUARE", "DOT", "OPEN_GRAPH", "CLOSE_GRAPH", "ARROW",
  "VBAR", "ASSIGN_POW", "ASSIGN_SHL", "ASSIGN_SHR", "ASSIGN_BXOR",
  "ASSIGN_BOR", "ASSIGN_BAND", "ASSIGN_MOD", "ASSIGN_DIV", "ASSIGN_MUL",
  "ASSIGN_SUB", "ASSIGN_ADD", "OP_EQ", "OP_AS", "OP_TO", "COMMA",
  "QUESTION", "OR", "AND", "NOT", "LE", "GE", "LT", "GT", "NEQ", "EEQ",
  "PROVIDES", "OP_NOTIN", "OP_IN", "DIESIS", "ATSIGN", "CAP_CAP",
  "VBAR_VBAR", "AMPER_AMPER", "MINUS", "PLUS", "PERCENT", "SLASH", "STAR",
  "POW", "SHR", "SHL", "CAP_XOROOB", "CAP_ISOOB", "CAP_DEOOB", "CAP_OOB",
  "CAP_EVAL", "TILDE", "NEG", "AMPER", "DECREMENT", "INCREMENT", "DOLLAR",
  "$accept", "input", "body", "line", "toplevel_statement",
  "INTNUM_WITH_MINUS", "load_statement", "statement", "base_statement",
  "assignment_def_list", "def_statement", "while_statement", "@1",
  "while_decl", "while_short_decl", "loop_statement", "@2",
  "loop_terminator", "if_statement", "@3", "if_decl", "if_short_decl",
  "elif_or_else", "@4", "else_decl", "elif_statement", "@5", "elif_decl",
  "statement_list", "break_statement", "continue_statement",
  "forin_statement", "@6", "@7", "forin_rest", "for_to_expr",
  "for_to_step_clause", "forin_statement_list", "forin_statement_elem",
  "fordot_statement", "self_print_statement", "outer_print_statement",
  "first_loop_block", "@8", "last_loop_block", "@9", "middle_loop_block",
  "@10", "switch_statement", "@11", "switch_decl", "case_list",
  "case_statement", "@12", "@13", "@14", "@15", "default_statement", "@16",
  "default_decl", "default_body", "case_expression_list", "case_element",
  "select_statement", "@17", "select_decl", "selcase_list",
  "selcase_statement", "@18", "@19", "@20", "@21",
  "selcase_expression_list", "selcase_element", "try_statement", "@22",
  "try_decl", "catch_statements", "catch_list", "catch_statement",
  "catch_decl", "catchcase_element_list", "catchcase_element",
  "raise_statement", "func_statement", "func_decl", "@23",
  "func_decl_short", "@24", "func_begin", "param_list", "param_symbol",
  "static_block", "@25", "@26", "static_decl", "static_short_decl",
  "launch_statement", "const_statement", "export_statement",
  "export_symbol_list", "import_statement", "import_symbol_list",
  "directive_statement", "directive_pair_list", "directive_pair",
  "class_decl", "@27", "class_def_inner", "class_param_list", "@28",
  "from_clause", "inherit_list", "inherit_token", "inherit_call",
  "class_statement_list", "class_statement", "init_decl", "@29",
  "property_decl", "enum_statement", "@30", "enum_statement_list",
  "enum_item_decl", "enum_item_terminator", "object_decl", "@31",
  "object_decl_inner", "object_statement_list", "object_statement",
  "global_statement", "@32", "global_symbol_list", "globalized_symbol",
  "return_statement", "const_atom", "atomic_symbol", "var_atom",
  "expression", "range_decl", "func_call", "@33", "nameless_func", "@34",
  "nameless_block", "@35", "nameless_func_decl_inner",
  "nameless_block_decl_inner", "innerfunc", "@36", "iif_expr",
  "array_decl", "dotarray_decl", "dict_decl", "expression_list",
  "listpar_expression_list", "listpar_comma", "symbol_list",
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
     365,   366,   367,   368,   369,   370
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   116,   117,   118,   118,   119,   119,   119,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   121,   121,
     122,   122,   122,   123,   123,   123,   123,   123,   123,   124,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   124,
     124,   124,   124,   124,   124,   124,   124,   124,   124,   125,
     125,   126,   126,   128,   127,   127,   129,   129,   130,   130,
     132,   131,   131,   131,   133,   133,   135,   134,   134,   136,
     136,   137,   137,   138,   138,   139,   138,   140,   140,   142,
     141,   143,   143,   144,   144,   145,   145,   146,   146,   146,
     148,   147,   149,   147,   147,   147,   150,   150,   151,   151,
     151,   152,   152,   152,   153,   153,   154,   154,   154,   154,
     155,   155,   156,   156,   156,   156,   156,   156,   157,   159,
     158,   158,   158,   161,   160,   160,   160,   163,   162,   162,
     162,   165,   164,   166,   166,   167,   167,   167,   168,   169,
     168,   170,   168,   171,   168,   172,   168,   173,   174,   173,
     175,   175,   175,   176,   176,   177,   177,   178,   178,   178,
     178,   178,   180,   179,   181,   181,   182,   182,   182,   183,
     184,   183,   185,   183,   186,   183,   187,   183,   188,   188,
     189,   189,   189,   190,   191,   190,   192,   192,   193,   193,
     194,   194,   195,   196,   196,   196,   196,   196,   197,   197,
     198,   198,   199,   199,   200,   200,   201,   202,   201,   201,
     203,   204,   203,   205,   206,   206,   206,   207,   208,   209,
     208,   210,   208,   211,   211,   212,   212,   213,   213,   214,
     214,   214,   215,   215,   215,   216,   216,   217,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   217,
     217,   218,   218,   219,   219,   220,   220,   221,   221,   221,
     223,   222,   224,   224,   225,   225,   226,   225,   227,   227,
     228,   228,   229,   230,   230,   230,   231,   231,   232,   232,
     232,   232,   234,   233,   235,   235,   237,   236,   238,   238,
     239,   239,   239,   240,   240,   242,   241,   243,   243,   244,
     244,   245,   245,   245,   245,   247,   246,   248,   248,   248,
     249,   250,   250,   250,   251,   251,   251,   251,   251,   251,
     252,   253,   253,   253,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   255,   255,   255,   255,   255,
     256,   256,   257,   256,   259,   258,   261,   260,   262,   262,
     262,   263,   263,   263,   265,   264,   266,   266,   266,   266,
     267,   267,   267,   268,   268,   268,   269,   269,   269,   270,
     270,   271,   271,   272,   272,   273,   273,   274,   274
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     3,     3,     3,     2,     2,
       4,     6,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       5,     3,     3,     0,     5,     2,     3,     3,     3,     3,
       0,     7,     3,     3,     0,     1,     0,     6,     2,     3,
       3,     3,     3,     0,     1,     0,     3,     2,     3,     0,
       4,     3,     3,     0,     2,     2,     3,     2,     3,     3,
       0,     6,     0,     6,     5,     3,     2,     4,     4,     4,
       3,     0,     2,     2,     0,     2,     1,     1,     1,     1,
       3,     3,     3,     2,     3,     2,     3,     3,     1,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     6,     3,
       3,     0,     6,     3,     3,     0,     2,     3,     1,     0,
       5,     0,     5,     0,     5,     0,     5,     0,     0,     3,
       0,     1,     2,     2,     2,     1,     3,     1,     1,     1,
       3,     1,     0,     6,     3,     3,     0,     2,     3,     1,
       0,     5,     0,     5,     0,     5,     0,     5,     1,     3,
       0,     1,     1,     3,     0,     6,     2,     3,     0,     1,
       1,     2,     2,     2,     4,     3,     5,     3,     1,     3,
       1,     1,     3,     3,     5,     2,     5,     0,     7,     3,
       5,     0,     6,     2,     0,     1,     3,     1,     0,     0,
       5,     0,     3,     2,     3,     2,     3,     3,     3,     5,
       5,     3,     2,     3,     3,     1,     3,     3,     5,     5,
       7,     7,     7,     7,     4,     4,     4,     4,     6,     6,
       3,     1,     3,     3,     3,     1,     3,     3,     3,     3,
       0,     7,     3,     2,     0,     3,     0,     5,     0,     2,
       1,     3,     2,     0,     2,     3,     0,     2,     1,     1,
       1,     1,     0,     7,     5,     4,     0,     7,     0,     2,
       1,     4,     2,     1,     1,     0,     7,     2,     2,     0,
       2,     1,     1,     1,     1,     0,     4,     1,     3,     3,
       1,     2,     3,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     2,     2,     2,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     2,     3,     2,     2,     2,     2,     3,     3,
       3,     3,     3,     3,     3,     2,     3,     3,     3,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     2,     1,     4,     5,     3,     1,
       1,     3,     5,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     4,     4,     5,     7,
       4,     3,     0,     6,     0,     6,     0,     6,     4,     3,
       2,     2,     2,     2,     0,     6,     5,     5,     4,     3,
       2,     3,     3,     2,     3,     3,     3,     3,     4,     1,
       3,     1,     3,     0,     1,     1,     3,     3,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     1,     0,    24,   317,   318,   320,   319,
     314,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   322,   323,     0,     0,     0,     0,     0,   305,     0,
       0,     0,     0,     0,     0,     0,   414,     0,     0,     0,
       0,   315,   316,   118,     0,     0,   406,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     5,     8,    14,    23,    32,    33,
      53,     0,    34,    38,    66,     0,    39,    40,    35,    46,
      47,    48,    36,   131,    37,   162,    41,   184,    42,    10,
     218,     0,     0,    45,    15,    16,    17,     9,    11,    13,
      12,    44,    43,   324,   321,   325,   429,   380,   371,   368,
     369,   370,   372,   375,   373,   379,     0,    28,     0,     6,
       0,   320,     0,     0,     0,   404,     0,     0,    85,     0,
      87,     0,     0,     0,     0,   435,     0,     0,     0,     0,
       0,     0,     0,   186,     0,     0,     0,     0,   260,     0,
     295,     0,   311,     0,     0,     0,     0,     0,     0,     0,
     371,     0,     0,     0,   232,   235,     0,     0,     0,     0,
       0,     0,     0,     0,   255,     0,   213,     0,     0,     0,
       0,   423,   431,     0,     0,    60,     0,   286,     0,     0,
     420,     0,   429,     0,     0,     0,   355,     0,   115,   429,
       0,   362,   361,   329,     0,   113,     0,   367,   366,   365,
     364,   363,   342,   327,   326,   328,   347,   345,   360,   359,
      83,     0,     0,     0,    55,    83,    68,   135,   166,    83,
       0,    83,   219,   221,   205,     0,     0,    29,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   346,   344,   374,     0,
       0,   330,    52,    51,     0,     0,    57,    59,    56,    58,
      86,    89,    88,    70,    72,    69,    71,    95,     0,     0,
       0,   134,   133,     7,   165,   164,   187,   183,   203,   202,
      27,     0,    26,     0,   313,   312,   310,     0,   307,    22,
      20,    21,   228,   227,   231,     0,   234,   233,     0,   250,
       0,     0,     0,     0,   237,     0,     0,   254,     0,   253,
       0,    25,     0,   214,   218,   218,   111,   110,   425,   424,
     434,     0,    63,    83,    62,     0,   394,   395,     0,   426,
       0,     0,   422,   421,     0,   427,     0,     0,   217,     0,
     215,   218,   117,   114,   116,   112,     0,     0,     0,     0,
       0,     0,   223,   225,     0,    83,     0,   209,   211,     0,
     401,     0,     0,     0,   378,   388,   392,   393,   391,   390,
     389,   387,   386,   385,   384,   383,   381,   419,     0,   354,
     353,   352,   351,   350,   349,   343,   348,   358,   357,   356,
     339,   338,   337,   332,   331,   335,   334,   333,   336,   341,
     340,     0,   430,     0,    49,    92,     0,   436,     0,    90,
       0,   214,   276,   268,     0,     0,     0,   299,   306,     0,
       0,     0,   236,   244,   246,     0,   247,     0,   245,     0,
       0,   252,    18,   257,   258,     0,   259,   256,   410,     0,
      83,    83,   432,     0,   288,   397,   396,     0,   437,   428,
       0,   413,   412,   411,     0,    83,     0,    84,     0,     0,
       0,    75,    74,    79,     0,   138,     0,     0,   136,     0,
     148,     0,   169,     0,     0,   167,     0,     0,   189,   190,
      83,   224,   226,     0,     0,   222,     0,   207,     0,   402,
     400,     0,   376,     0,   418,     0,    30,     0,     0,     0,
       0,    94,     0,   263,     0,     0,     0,   298,   273,   269,
     270,   297,     0,   309,   308,   230,   229,     0,     0,   238,
       0,     0,   239,     0,     0,    19,   409,     0,     0,     0,
      64,     0,     0,   398,     0,   216,     0,    54,     0,    77,
       0,     0,     0,    83,    83,   137,     0,   161,   159,   157,
     158,     0,   155,   152,     0,     0,   168,     0,   181,   182,
       0,   178,     0,     0,   193,   200,   201,     0,     0,   198,
       0,   191,     0,   204,     0,     0,     0,   206,   210,     0,
     377,   382,   417,   416,     0,    50,     0,     0,    93,   100,
       0,    91,   266,   265,   278,     0,     0,     0,     0,     0,
     279,   277,   281,   280,   262,     0,   272,     0,   301,     0,
     302,   304,   303,   300,   248,   249,     0,     0,     0,     0,
     408,   405,   415,     0,    65,   290,     0,     0,   289,     0,
     438,   407,    78,    82,    81,    67,     0,     0,   143,   145,
       0,   139,   141,     0,   132,    83,     0,   149,   174,   176,
     170,   172,   180,   163,   197,     0,   195,     0,     0,   185,
     220,   212,     0,   403,    31,     0,     0,     0,   106,     0,
       0,   107,   108,   109,    96,    99,     0,    98,     0,     0,
     261,   282,     0,   274,     0,   271,   296,   240,   242,   241,
     243,    61,   293,     0,   294,   292,   287,   399,    80,    83,
       0,   160,    83,     0,   156,     0,   154,    83,     0,    83,
       0,   179,   194,   199,     0,   208,     0,   119,     0,     0,
     123,     0,     0,   127,     0,     0,   105,   103,   102,   267,
       0,   218,     0,   275,     0,     0,   146,     0,   142,     0,
     177,     0,   173,   196,   122,    83,   121,   126,    83,   125,
     130,    83,   129,    97,   285,    83,     0,   291,     0,     0,
       0,     0,   284,     0,     0,     0,     0,   120,   124,   128,
     283
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    63,    64,   580,    65,   487,    67,   122,
      68,    69,   220,    70,    71,    72,   353,   653,    73,   225,
      74,    75,   490,   573,   491,   492,   574,   493,   376,    76,
      77,    78,   532,   529,   618,   435,   707,   699,   700,    79,
      80,    81,   701,   775,   702,   778,   703,   781,    82,   227,
      83,   378,   498,   732,   733,   729,   730,   499,   585,   500,
     677,   581,   582,    84,   228,    85,   379,   505,   739,   740,
     737,   738,   590,   591,    86,   229,    87,   507,   508,   509,
     510,   598,   599,    88,    89,    90,   606,    91,   516,    92,
     369,   370,   231,   385,   386,   232,   233,    93,    94,    95,
     166,    96,   170,    97,   173,   174,    98,   311,   442,   443,
     708,   446,   539,   540,   636,   535,   631,   632,   761,   633,
      99,   355,   561,   658,   725,   100,   313,   447,   542,   643,
     101,   154,   317,   318,   102,   103,   104,   105,   106,   107,
     108,   609,   109,   177,   110,   195,   344,   371,   111,   178,
     112,   113,   114,   115,   116,   183,   351,   136,   194
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -419
static const yytype_int16 yypact[] =
{
    -419,    17,   765,  -419,    25,  -419,  -419,  -419,   -16,  -419,
    -419,    88,   270,    56,   507,   442,  3583,   278,  3655,   128,
    3727,  -419,  -419,   177,  3799,   310,   432,  3367,  -419,   537,
    3871,   449,   520,   281,   506,    80,  -419,  3943,  5397,   221,
     154,  -419,  -419,  -419,  5757,  5253,  -419,  5757,  3439,  5757,
    5757,  5757,  3511,  5757,  5757,  5757,  5757,  5757,  5757,   458,
    5757,  5757,   513,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  3295,  -419,  -419,  -419,  3295,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
     171,  3295,    65,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  4744,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,   150,  -419,  5757,  -419,
     183,  -419,    81,   189,   179,  -419,  4569,   305,  -419,   319,
    -419,   323,   247,  4642,   329,   301,     2,   423,  4795,   427,
     433,  4846,   441,  -419,  3295,   444,  4897,   456,  -419,   460,
    -419,   466,  -419,  4948,   484,   496,   531,   543,   551,  6298,
     556,   560,   462,   562,  -419,  -419,   110,   563,    55,   534,
     140,   566,   498,   112,  -419,   568,  -419,   127,   127,   571,
    4999,  -419,  6298,   475,   572,  -419,  3295,  -419,  5990,  5469,
    -419,   519,  5817,    33,    78,    96,  6494,   574,  -419,  6298,
     115,   261,   261,   265,   583,  -419,   120,   265,   265,   265,
     265,   265,   265,  -419,  -419,  -419,   265,   265,  -419,  -419,
    -419,   590,   591,   131,  -419,  -419,  -419,  -419,  -419,  -419,
     303,  -419,  -419,  -419,  -419,   592,   121,  -419,  5541,  5325,
     587,  5757,  5757,  5757,  5757,  5757,  5757,  5757,  5757,  5757,
    5757,  5757,  5757,  4015,  5757,  5757,  5757,  5757,  5757,  5757,
    5757,  5757,   595,  5757,  5757,  5757,  5757,  5757,  5757,  5757,
    5757,  5757,  5757,  5757,  5757,  5757,  -419,  -419,  -419,  5757,
    5757,  6298,  -419,  -419,   596,  5757,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  5757,   596,
    4087,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,   282,  -419,   497,  -419,  -419,  -419,   194,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,   453,  -419,  -419,   598,  -419,
     593,    18,    19,   602,  -419,   555,   601,  -419,    47,  -419,
     604,  -419,   606,   605,   171,   171,  -419,  -419,  -419,  -419,
    -419,  5757,  -419,  -419,  -419,   609,  -419,  -419,  6041,  -419,
    5613,  5757,  -419,  -419,   561,  -419,  5757,   536,  -419,    10,
    -419,   171,  -419,  -419,  -419,  -419,  1800,  1455,   306,   465,
    1570,   287,  -419,  -419,  1915,  -419,  3295,  -419,  -419,   136,
    -419,   211,  5757,  5878,  -419,  6298,  6298,  6298,  6298,  6298,
    6298,  6298,  6298,  6298,  6298,  6298,   541,  -419,  4496,  6445,
    6494,  5729,  5729,  5729,  5729,  5729,  5729,  -419,   261,   261,
     603,   603,  1420,   269,   269,   383,   383,   383,   390,   265,
     265,  4693,  6347,   542,  6298,  -419,  6092,  -419,   630,  6298,
     631,   605,  -419,   614,   645,   643,   647,  -419,  -419,   524,
     648,   653,  -419,  -419,  -419,   652,  -419,   654,  -419,    16,
      49,  -419,  -419,  -419,  -419,   659,  -419,  -419,  -419,   214,
    -419,  -419,  6298,  2030,  -419,  -419,  -419,  5939,  6298,  -419,
    6145,  -419,  -419,  -419,   605,  -419,   661,  -419,   544,  4159,
     656,  -419,  -419,  -419,   663,  -419,    70,   309,  -419,   658,
    -419,   665,  -419,   274,   660,  -419,    64,   662,   641,  -419,
    -419,  -419,  -419,   669,  2145,  -419,   618,  -419,   300,  -419,
    -419,  6196,  -419,  5757,  -419,  4231,  -419,  5757,  5757,   337,
    4303,  -419,   337,  -419,   215,   296,   672,  -419,   620,   599,
    -419,  -419,   533,  -419,  -419,  -419,  -419,   675,   676,  -419,
     674,   677,  -419,   678,   680,  -419,  -419,   679,  2260,  2375,
    5757,   148,  5757,  -419,  5757,  -419,  2490,  -419,   684,  -419,
     685,  5050,   686,  -419,  -419,  -419,   407,  -419,  -419,  -419,
     613,    52,  -419,  -419,   688,   428,  -419,   431,  -419,  -419,
     101,  -419,   689,   690,  -419,  -419,  -419,   596,     7,  -419,
     692,  -419,  1685,  -419,   693,   637,   655,  -419,  -419,   657,
    -419,   635,  -419,  6396,   216,  6298,   880,  3295,  -419,  -419,
    4435,  -419,  -419,  -419,  -419,   633,   706,   712,   714,   716,
    -419,  -419,  -419,  -419,  -419,  5685,  -419,   643,  -419,   715,
    -419,  -419,  -419,  -419,  -419,  -419,   720,   721,   722,   723,
    -419,  -419,  -419,   724,  6298,  -419,    72,   726,  -419,  6247,
    6298,  -419,  -419,  -419,  -419,  -419,  2605,  1455,  -419,  -419,
       3,  -419,  -419,    23,  -419,  -419,  3295,  -419,  -419,  -419,
    -419,  -419,   545,  -419,  -419,   727,  -419,   554,   596,  -419,
    -419,  -419,   729,  -419,  -419,   333,   384,   434,  -419,   725,
     880,  -419,  -419,  -419,  -419,  -419,  4375,  -419,   682,  5757,
    -419,  -419,   664,  -419,   288,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,   293,  -419,  -419,  -419,  -419,  -419,  -419,
    3295,  -419,  -419,  3295,  -419,  2720,  -419,  -419,  3295,  -419,
    3295,  -419,  -419,  -419,   730,  -419,   732,  -419,  3295,   737,
    -419,  3295,   738,  -419,  3295,   739,  -419,  -419,  6298,  -419,
    5101,   171,  5757,  -419,   217,   995,  -419,  1110,  -419,  1225,
    -419,  1340,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  5152,  -419,  2835,  2950,
    3065,  3180,  -419,   740,   741,   742,   744,  -419,  -419,  -419,
    -419
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -419,  -419,  -419,  -419,  -419,  -333,  -419,    -2,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,    87,  -419,  -419,  -419,  -419,  -419,   -18,  -419,
    -419,  -419,  -419,  -419,   223,  -419,  -419,    48,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,  -419,  -419,   377,  -419,  -419,
    -419,  -419,    85,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,    77,  -419,  -419,  -419,  -419,  -419,   253,
    -419,  -419,    75,  -419,  -418,  -419,  -419,  -419,  -419,  -419,
    -233,   280,  -321,  -419,  -419,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,  -419,   445,  -419,  -419,  -419,  -419,
    -419,   338,  -419,   145,  -419,  -419,  -419,   242,  -419,   244,
    -419,  -419,  -419,  -419,    30,  -419,  -419,  -419,  -419,  -419,
    -419,  -419,  -419,   340,  -419,  -323,    -3,  -419,   -12,    -7,
     767,  -419,  -419,  -419,  -419,  -419,   623,  -419,  -419,  -419,
    -419,  -419,  -419,  -419,   -32,  -419,  -419,  -419,  -419
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -434
static const yytype_int16 yytable[] =
{
      66,   126,   451,   389,   133,   466,   138,   462,   141,   123,
     686,   482,   146,   193,   135,   153,   200,     3,   159,   549,
     206,   454,   456,   470,   471,   180,   182,   462,   117,   577,
     578,   579,   188,   192,   362,   196,   199,   201,   202,   203,
     199,   207,   208,   209,   210,   211,   212,   118,   216,   217,
     485,   462,   552,   463,   464,   671,   330,   124,  -251,   219,
       6,     7,     8,     9,    10,   593,   235,   594,   595,   224,
     596,   576,   483,   226,   462,   722,   577,   578,   579,   364,
     299,   175,    21,    22,   283,   687,   176,  -251,   484,   234,
     363,   119,   550,   300,   455,   457,   672,   367,   688,   278,
     465,   125,   368,    36,   680,    38,   281,   551,    41,    42,
     469,   280,    44,   327,    45,   339,    46,   630,   373,   278,
     465,   236,   388,   375,   640,   553,   278,   368,   342,   139,
     673,   278,   175,  -251,   278,   365,  -404,   517,    47,   278,
     554,   333,   307,   334,   465,   681,   278,   723,    49,    50,
     724,   655,   278,    51,   656,   597,   366,   657,  -214,   284,
     187,    53,    54,    55,    56,    57,    58,   465,    59,    60,
      61,    62,   335,   278,  -214,   278,  -214,   358,   142,   682,
     143,   278,   286,   343,   354,   278,   282,  -404,   328,   278,
     340,   518,   278,   280,   278,   278,   278,   448,   280,  -214,
     278,   278,   278,   278,   278,   278,   391,   377,   534,   278,
     278,   380,   519,   384,   484,   556,   622,   230,   336,   694,
     722,   144,   184,   287,   185,   279,   199,   393,   280,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     406,   408,   409,   410,   411,   412,   413,   414,   415,   416,
     293,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   285,   186,   520,   431,   432,   557,
     623,   120,   449,   434,   278,   587,   121,  -180,   588,   134,
     589,   433,   167,   440,   121,  -264,   436,   168,   439,   280,
     511,   294,   484,   484,   280,   724,   437,     6,     7,   624,
       9,    10,   625,   607,   381,   626,   382,   494,   290,   495,
     583,   147,  -151,   169,  -264,  -147,   148,   238,  -180,   239,
     240,   238,   291,   239,   240,   238,   292,   239,   240,   496,
     497,   512,   297,   627,   746,   473,   747,   731,   441,   472,
     616,   628,   629,   763,   608,    41,    42,   383,   477,   478,
    -150,   278,  -180,  -151,   480,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   280,   514,   270,   271,
     272,   273,   274,   275,   276,   277,   298,   748,   276,   277,
     521,   617,   276,   277,   515,   749,   278,   750,   278,   278,
     278,   278,   278,   278,   278,   278,   278,   278,   278,   278,
     764,   278,   278,   278,   278,   278,   278,   278,   278,   278,
     668,   278,   278,   278,   278,   278,   278,   278,   278,   278,
     278,   278,   278,   278,   278,   278,   301,   278,   751,   278,
     303,   675,   278,   149,   678,   752,   304,   753,   150,   238,
     785,   239,   240,   129,   306,   130,   238,   308,   239,   240,
     161,   669,   558,   559,   450,   162,   131,     6,     7,   310,
       9,    10,   213,   312,   214,   278,   501,   566,   502,   314,
     278,   278,   676,   278,  -147,   679,   348,   571,   754,  -433,
    -433,  -433,  -433,  -433,   215,   273,   274,   275,   503,   497,
     316,   611,   602,   274,   275,   614,   276,   277,   444,   319,
    -268,  -433,  -433,   276,   277,    41,    42,   171,   127,  -150,
     128,   199,   172,   613,   278,   199,   615,   218,   620,   121,
    -433,   163,  -433,   164,  -433,   543,   165,  -433,  -433,   445,
     316,  -433,   349,  -433,   320,  -433,   638,   325,   155,   625,
     331,   332,   639,   156,   157,   568,   321,   569,   654,   588,
     659,   589,   660,   350,   322,   666,   667,  -433,   595,   323,
     596,   459,   460,   324,   278,   326,   329,  -433,  -433,   337,
     627,   341,  -433,   338,   346,   352,   359,   372,   628,   629,
    -433,  -433,  -433,  -433,  -433,  -433,   374,  -433,  -433,  -433,
    -433,   147,   149,   394,   685,   387,   453,   238,   481,   239,
     240,   417,   121,   714,   452,   458,   278,   461,   278,   468,
     172,   368,   474,   278,   698,   704,   252,   528,   479,   523,
     253,   254,   255,   199,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   531,   533,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,   445,   278,   537,   538,
     541,   545,   278,   278,   276,   277,   546,   735,   547,   238,
     548,   239,   240,   555,   567,   572,   575,   584,   586,   592,
     506,   600,   603,   605,   736,   634,   635,   637,   644,   645,
     646,   691,   650,   647,   648,   744,   649,   662,   663,   665,
     670,   674,   683,   684,   758,   689,   690,   760,   698,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   709,   710,
     692,   765,   693,   280,   767,   711,   276,   277,   716,   769,
     176,   771,   712,   717,   718,   719,   720,   721,   766,   726,
     742,   768,   745,   773,   755,   774,   770,   759,   772,   762,
     777,   780,   783,   797,   798,   799,   776,   800,   756,   779,
     786,   278,   782,   278,   728,   621,   504,   788,   734,   741,
     789,   601,   743,   790,   565,    -2,     4,   791,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,   278,
      16,   536,   715,    17,   641,   467,   642,    18,    19,   544,
      20,    21,    22,    23,   787,    24,    25,   160,    26,    27,
      28,   345,     0,    29,    30,    31,    32,    33,    34,     0,
      35,     0,    36,    37,    38,    39,    40,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -104,
      12,    13,    14,    15,     0,    16,     0,     0,    17,   695,
     696,   697,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -144,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -144,  -144,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,  -144,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -140,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -140,  -140,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,  -140,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -175,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,  -175,  -175,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,  -175,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -171,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -171,  -171,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,  -171,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   -73,    12,    13,    14,    15,     0,
      16,   488,   489,    17,     0,     0,   238,    18,   239,   240,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,     0,     0,     0,
       0,     0,     0,   276,   277,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -188,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,   506,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,  -192,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,  -192,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   486,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   513,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   560,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   604,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   651,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   652,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,     0,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,   661,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   -76,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -153,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   793,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   794,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,   795,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   796,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   221,     0,   222,    27,    28,     0,     0,     0,    30,
       0,     0,     0,     0,     0,   223,     0,    36,    37,    38,
      39,     0,    41,    42,    43,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,     0,     0,     0,    48,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,     0,     0,
       0,     0,     0,    52,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,     4,     0,     5,     6,
       7,     8,     9,    10,     0,    12,    13,    14,    15,     0,
      16,     0,     0,    17,     0,     0,     0,    18,     0,     0,
      20,    21,    22,    23,     0,    24,   221,     0,   222,    27,
      28,     0,     0,     0,    30,     0,     0,     0,     0,     0,
     223,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   151,     0,
     152,     6,     7,     8,     9,    10,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     0,   125,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     197,     0,   198,     6,     7,     8,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     0,   125,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   204,     0,   205,     6,     7,     8,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     0,   125,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   132,     0,     0,     6,     7,     8,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     0,   125,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   137,     0,     0,     6,
       7,     8,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     0,
     125,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   140,     0,
       0,     6,     7,     8,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     0,   125,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     145,     0,     0,     6,     7,     8,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     0,   125,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   158,     0,     0,     6,     7,     8,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     0,   125,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   179,     0,     0,     6,     7,     8,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     0,   125,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   407,     0,     0,     6,
       7,     8,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     0,
     125,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   438,     0,
       0,     6,     7,     8,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     0,   125,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     570,     0,     0,     6,     7,     8,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     0,   125,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   612,     0,     0,     6,     7,     8,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     0,   125,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   619,     0,     0,     6,     7,     8,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     0,   125,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   757,     0,     0,     6,
       7,     8,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     0,
     125,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,    44,     0,    45,     0,    46,   705,     0,  -101,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,     0,  -101,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   238,     0,   239,   240,     0,     0,   524,     0,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,     0,     0,   706,   253,   254,   255,     0,   256,   257,
     258,   259,   260,   261,   262,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     525,     0,     0,     0,     0,     0,     0,     0,   276,   277,
       0,     0,   238,     0,   239,   240,     0,     0,     0,     0,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   288,     0,     0,   253,   254,   255,     0,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,     0,     0,     0,     0,     0,     0,   276,
     277,     0,     0,   289,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   238,     0,   239,   240,     0,
       0,     0,     0,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   295,     0,     0,   253,   254,
     255,     0,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,     0,     0,     0,     0,
       0,     0,   276,   277,     0,     0,   296,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   526,     0,   238,     0,
     239,   240,     0,     0,     0,     0,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,     0,     0,
       0,   253,   254,   255,     0,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,   237,     0,   238,
       0,   239,   240,     0,     0,   276,   277,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   252,     0,
       0,   527,   253,   254,   255,     0,   256,   257,   258,   259,
     260,   261,   262,   263,   264,     0,     0,   265,   266,   267,
     268,   269,   270,   271,   272,   273,   274,   275,   302,     0,
     238,     0,   239,   240,     0,     0,   276,   277,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   251,   252,
       0,     0,     0,   253,   254,   255,     0,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,   305,
       0,   238,     0,   239,   240,     0,     0,   276,   277,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,     0,     0,     0,   253,   254,   255,     0,   256,   257,
     258,   259,   260,   261,   262,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     309,     0,   238,     0,   239,   240,     0,     0,   276,   277,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,     0,     0,     0,   253,   254,   255,     0,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   315,     0,   238,     0,   239,   240,     0,     0,   276,
     277,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,     0,     0,     0,   253,   254,   255,     0,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   347,     0,   238,     0,   239,   240,     0,     0,
     276,   277,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,     0,     0,     0,   253,   254,   255,
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   664,     0,   238,     0,   239,   240,     0,
       0,   276,   277,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,     0,     0,     0,   253,   254,
     255,     0,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   784,     0,   238,     0,   239,   240,
       0,     0,   276,   277,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,     0,     0,     0,   253,
     254,   255,     0,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   792,     0,   238,     0,   239,
     240,     0,     0,   276,   277,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,     0,     0,     0,
     253,   254,   255,     0,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,     0,     0,   238,     0,
     239,   240,     0,     0,   276,   277,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,     0,     0,
       0,   253,   254,   255,     0,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     6,     7,     8,
       9,    10,     0,     0,     0,   276,   277,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    21,
      22,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   189,   125,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
     190,    45,     0,    46,     0,   191,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   189,
     125,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,     8,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,   392,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     0,   125,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,   181,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,     8,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     0,   125,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,   357,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,     8,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     0,   125,     0,    36,     0,
      38,     0,     0,    41,    42,     0,   390,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     0,   125,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
     476,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     0,
     125,     0,    36,     0,    38,     0,     0,    41,    42,     0,
     713,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,     8,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,   238,     0,   239,   240,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     0,   125,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,     0,     0,     0,     0,    47,
       0,     0,   276,   277,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
       0,   360,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   238,     0,   239,   240,     0,     0,   361,
       0,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,     0,     0,     0,   253,   254,   255,     0,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   360,     0,     0,     0,     0,     0,     0,     0,
     276,   277,     0,     0,   238,   522,   239,   240,     0,     0,
       0,     0,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,     0,     0,     0,   253,   254,   255,
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   562,     0,     0,     0,     0,     0,     0,
       0,   276,   277,     0,     0,   238,   563,   239,   240,     0,
       0,     0,     0,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,     0,     0,     0,   253,   254,
     255,     0,   256,   257,   258,   259,   260,   261,   262,   263,
     264,     0,     0,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,     0,   356,   238,     0,   239,   240,
       0,     0,   276,   277,   241,   242,   243,   244,   245,   246,
     247,   248,   249,   250,   251,   252,     0,     0,     0,   253,
     254,   255,     0,   256,   257,   258,   259,   260,   261,   262,
     263,   264,     0,     0,   265,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,     0,     0,   238,   475,   239,
     240,     0,     0,   276,   277,   241,   242,   243,   244,   245,
     246,   247,   248,   249,   250,   251,   252,     0,     0,     0,
     253,   254,   255,     0,   256,   257,   258,   259,   260,   261,
     262,   263,   264,     0,     0,   265,   266,   267,   268,   269,
     270,   271,   272,   273,   274,   275,     0,     0,   238,     0,
     239,   240,     0,     0,   276,   277,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,     0,   530,
       0,   253,   254,   255,     0,   256,   257,   258,   259,   260,
     261,   262,   263,   264,     0,     0,   265,   266,   267,   268,
     269,   270,   271,   272,   273,   274,   275,     0,     0,     0,
       0,   238,     0,   239,   240,   276,   277,   564,     0,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,     0,     0,     0,   253,   254,   255,     0,   256,   257,
     258,   259,   260,   261,   262,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
       0,     0,   238,   610,   239,   240,     0,     0,   276,   277,
     241,   242,   243,   244,   245,   246,   247,   248,   249,   250,
     251,   252,     0,     0,     0,   253,   254,   255,     0,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,     0,     0,   238,   727,   239,   240,     0,     0,   276,
     277,   241,   242,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,     0,     0,     0,   253,   254,   255,     0,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,     0,     0,   238,     0,   239,   240,     0,     0,
     276,   277,   241,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,     0,     0,     0,   253,   254,   255,
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
       0,     0,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   275,   238,     0,   239,   240,     0,     0,     0,
       0,   276,   277,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   253,   254,   255,     0,
     256,   257,   258,   259,   260,   261,   262,   263,   264,     0,
       0,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   238,     0,   239,   240,     0,     0,     0,     0,
     276,   277,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   254,   255,     0,   256,
     257,   258,   259,   260,   261,   262,   263,   264,     0,     0,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   238,     0,   239,   240,     0,     0,     0,     0,   276,
     277,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   255,     0,   256,   257,
     258,   259,   260,   261,   262,   263,   264,     0,     0,   265,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     238,     0,   239,   240,     0,     0,     0,     0,   276,   277,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   256,   257,   258,
     259,   260,   261,   262,   263,   264,     0,     0,   265,   266,
     267,   268,   269,   270,   271,   272,   273,   274,   275,     0,
       0,     0,     0,     0,     0,     0,     0,   276,   277
};

static const yytype_int16 yycheck[] =
{
       2,    13,   325,   236,    16,   338,    18,     4,    20,    12,
       3,     1,    24,    45,    17,    27,    48,     0,    30,     3,
      52,     3,     3,   344,   345,    37,    38,     4,     3,     6,
       7,     8,    44,    45,     1,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    63,    60,    61,
     371,     4,     3,     6,     7,     3,     1,     1,     3,    62,
       4,     5,     6,     7,     8,     1,     1,     3,     4,    71,
       6,     1,    62,    75,     4,     3,     6,     7,     8,     1,
      78,     1,    26,    27,     3,    78,     6,    32,    78,    91,
      57,     3,    76,    91,    76,    76,    44,     1,    91,   106,
      97,    45,     6,    47,     3,    49,   118,    91,    52,    53,
     343,    78,    56,     3,    58,     3,    60,   535,     3,   126,
      97,    56,     1,     3,   542,    76,   133,     6,     1,     1,
      78,   138,     1,    78,   141,    57,    56,     1,    82,   146,
      91,     1,   144,     3,    97,    44,   153,    75,    92,    93,
      78,     3,   159,    97,     6,    91,    78,     9,    62,    78,
       6,   105,   106,   107,   108,   109,   110,    97,   112,   113,
     114,   115,    32,   180,    78,   182,    55,   189,     1,    78,
       3,   188,     3,    56,   186,   192,     3,    56,    78,   196,
      78,    55,   199,    78,   201,   202,   203,     3,    78,    78,
     207,   208,   209,   210,   211,   212,   238,   225,   441,   216,
     217,   229,     1,   231,    78,     1,     1,    46,    78,     3,
       3,    44,     1,    44,     3,    75,   238,   239,    78,   241,
     242,   243,   244,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
       3,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   274,   275,    75,    44,    55,   279,   280,    55,
      55,     1,    78,   285,   281,     1,     6,     3,     4,     1,
       6,   284,     1,     1,     6,     3,   298,     6,   300,    78,
       3,    44,    78,    78,    78,    78,   299,     4,     5,     3,
       7,     8,     6,     3,     1,     9,     3,     1,     3,     3,
       1,     1,     3,    32,    32,     9,     6,    56,    44,    58,
      59,    56,     3,    58,    59,    56,     3,    58,    59,    23,
      24,    44,     3,    37,     1,   353,     3,   670,    56,   351,
       3,    45,    46,    55,    44,    52,    53,    44,   360,   361,
      44,   358,    78,    44,   366,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    78,   385,    99,   100,
     101,   102,   103,   104,   113,   114,    75,    44,   113,   114,
     392,    44,   113,   114,   386,     1,   393,     3,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,   406,
     723,   408,   409,   410,   411,   412,   413,   414,   415,   416,
       3,   418,   419,   420,   421,   422,   423,   424,   425,   426,
     427,   428,   429,   430,   431,   432,     3,   434,    44,   436,
       3,     3,   439,     1,     3,     1,     3,     3,     6,    56,
     761,    58,    59,     1,     3,     3,    56,     3,    58,    59,
       1,    44,   470,   471,     1,     6,    14,     4,     5,     3,
       7,     8,     4,     3,     6,   472,     1,   485,     3,     3,
     477,   478,    44,   480,     9,    44,     1,   489,    44,     4,
       5,     6,     7,     8,    26,   102,   103,   104,    23,    24,
       6,   523,   510,   103,   104,   527,   113,   114,     1,     3,
       3,    26,    27,   113,   114,    52,    53,     1,     1,    44,
       3,   523,     6,   525,   521,   527,   528,     4,   530,     6,
      45,     1,    47,     3,    49,     1,     6,    52,    53,    32,
       6,    56,    57,    58,     3,    60,     3,    75,     1,     6,
       6,     7,     9,     6,     7,     1,     3,     3,   560,     4,
     562,     6,   564,    78,     3,   573,   574,    82,     4,     3,
       6,     6,     7,     3,   571,     3,     3,    92,    93,     3,
      37,     3,    97,    75,     3,     3,    57,     3,    45,    46,
     105,   106,   107,   108,   109,   110,     3,   112,   113,   114,
     115,     1,     1,     6,   597,     3,     3,    56,    62,    58,
      59,     6,     6,   635,     6,     3,   613,     6,   615,     3,
       6,     6,     3,   620,   616,   617,    75,    75,    57,    78,
      79,    80,    81,   635,    83,    84,    85,    86,    87,    88,
      89,    90,    91,     3,     3,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    32,   654,     3,     6,
       3,     3,   659,   660,   113,   114,     3,   675,     6,    56,
       6,    58,    59,     4,     3,     9,     3,     9,     3,     9,
      29,     9,     3,    55,   676,     3,    56,    78,     3,     3,
       6,    44,     3,     6,     6,   688,     6,     3,     3,     3,
      77,     3,     3,     3,   706,     3,     3,   709,   700,    96,
      97,    98,    99,   100,   101,   102,   103,   104,    75,     3,
      55,   729,    55,    78,   732,     3,   113,   114,     3,   737,
       6,   739,     6,     3,     3,     3,     3,     3,   730,     3,
       3,   733,     3,     3,     9,     3,   738,    55,   740,    75,
       3,     3,     3,     3,     3,     3,   748,     3,   700,   751,
     762,   758,   754,   760,   667,   532,   379,   775,   673,   682,
     778,   508,   687,   781,   484,     0,     1,   785,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,   786,
      15,   443,   637,    18,   542,   340,   542,    22,    23,   449,
      25,    26,    27,    28,   764,    30,    31,    30,    33,    34,
      35,   178,    -1,    38,    39,    40,    41,    42,    43,    -1,
      45,    -1,    47,    48,    49,    50,    51,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    19,
      20,    21,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    44,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    44,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    44,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    16,    17,    18,    -1,    -1,    56,    22,    58,    59,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   113,   114,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    29,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    29,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,    -1,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    61,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    -1,    39,
      -1,    -1,    -1,    -1,    -1,    45,    -1,    47,    48,    49,
      50,    -1,    52,    53,    54,    -1,    56,    -1,    58,    -1,
      60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    -1,    -1,    -1,    86,    -1,    -1,    -1,
      -1,    -1,    92,    93,    -1,    -1,    -1,    97,    -1,    -1,
      -1,    -1,    -1,   103,    -1,   105,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,     1,    -1,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    -1,
      15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,
      25,    26,    27,    28,    -1,    30,    31,    -1,    33,    34,
      35,    -1,    -1,    -1,    39,    -1,    -1,    -1,    -1,    -1,
      45,    -1,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       3,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    86,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,   103,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    -1,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    -1,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     3,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    -1,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    -1,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    -1,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    -1,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    -1,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    -1,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    -1,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    -1,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,    -1,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    -1,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    -1,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    -1,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    -1,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    -1,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    -1,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      -1,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    -1,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,     1,    -1,     3,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,    -1,    44,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    56,    -1,    58,    59,    -1,    -1,     1,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    78,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114,
      -1,    -1,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,     3,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,
     114,    -1,    -1,    44,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    56,    -1,    58,    59,    -1,
      -1,    -1,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,     3,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   113,   114,    -1,    -1,    44,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,    56,    -1,
      58,    59,    -1,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,     3,    -1,    56,
      -1,    58,    59,    -1,    -1,   113,   114,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    -1,
      -1,    78,    79,    80,    81,    -1,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    -1,    -1,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,   104,     3,    -1,
      56,    -1,    58,    59,    -1,    -1,   113,   114,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
      -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,     3,
      -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
       3,    -1,    56,    -1,    58,    59,    -1,    -1,   113,   114,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,     3,    -1,    56,    -1,    58,    59,    -1,    -1,   113,
     114,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,     3,    -1,    56,    -1,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,     3,    -1,    56,    -1,    58,    59,    -1,
      -1,   113,   114,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,     3,    -1,    56,    -1,    58,    59,
      -1,    -1,   113,   114,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,     3,    -1,    56,    -1,    58,
      59,    -1,    -1,   113,   114,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    -1,    56,    -1,
      58,    59,    -1,    -1,   113,   114,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    -1,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,     4,     5,     6,
       7,     8,    -1,    -1,    -1,   113,   114,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    26,
      27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      57,    58,    -1,    60,    -1,    62,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    44,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      -1,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    -1,   101,    -1,    -1,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    -1,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    57,    58,    -1,    60,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    26,    27,    -1,    -1,    -1,
      -1,    -1,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    -1,    45,    -1,    47,    -1,    49,    -1,
      -1,    52,    53,    -1,    -1,    56,    57,    58,    -1,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    92,    93,    -1,    -1,    -1,    97,    26,    27,    -1,
      -1,    -1,    -1,    -1,   105,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,    -1,    45,    -1,    47,    -1,
      49,    -1,    -1,    52,    53,    -1,    55,    56,    -1,    58,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    82,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    92,    93,    -1,    -1,    -1,    97,    26,
      27,    -1,    -1,    -1,    -1,    -1,   105,   106,   107,   108,
     109,   110,    -1,   112,   113,   114,   115,    -1,    45,    -1,
      47,    -1,    49,    -1,    -1,    52,    53,    -1,    -1,    56,
      57,    58,    -1,    60,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    82,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    92,    93,    -1,    -1,    -1,
      97,    26,    27,    -1,    -1,    -1,    -1,    -1,   105,   106,
     107,   108,   109,   110,    -1,   112,   113,   114,   115,    -1,
      45,    -1,    47,    -1,    49,    -1,    -1,    52,    53,    -1,
      55,    56,    -1,    58,    -1,    60,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    82,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    92,    93,    -1,
      -1,    -1,    97,    26,    27,    56,    -1,    58,    59,    -1,
     105,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,    -1,    45,    -1,    47,    -1,    49,    -1,    -1,    52,
      53,    -1,    -1,    56,    -1,    58,    -1,    60,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,   113,   114,    -1,    -1,    -1,    -1,    -1,    92,
      93,    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,
      -1,    44,   105,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,    56,    -1,    58,    59,    -1,    -1,    62,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     113,   114,    -1,    -1,    56,    57,    58,    59,    -1,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    44,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   113,   114,    -1,    -1,    56,    57,    58,    59,    -1,
      -1,    -1,    -1,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    -1,    -1,    -1,    79,    80,
      81,    -1,    83,    84,    85,    86,    87,    88,    89,    90,
      91,    -1,    -1,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,   104,    -1,    55,    56,    -1,    58,    59,
      -1,    -1,   113,   114,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    -1,    -1,    -1,    79,
      80,    81,    -1,    83,    84,    85,    86,    87,    88,    89,
      90,    91,    -1,    -1,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,   104,    -1,    -1,    56,    57,    58,
      59,    -1,    -1,   113,   114,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    -1,    -1,    -1,
      79,    80,    81,    -1,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    -1,    -1,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,    -1,    -1,    56,    -1,
      58,    59,    -1,    -1,   113,   114,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    75,    -1,    77,
      -1,    79,    80,    81,    -1,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    -1,    -1,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,   104,    -1,    -1,    -1,
      -1,    56,    -1,    58,    59,   113,   114,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      -1,    -1,    56,    57,    58,    59,    -1,    -1,   113,   114,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    -1,    -1,    56,    57,    58,    59,    -1,    -1,   113,
     114,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    -1,    -1,    56,    -1,    58,    59,    -1,    -1,
     113,   114,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    -1,    -1,    -1,    79,    80,    81,
      -1,    83,    84,    85,    86,    87,    88,    89,    90,    91,
      -1,    -1,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,   104,    56,    -1,    58,    59,    -1,    -1,    -1,
      -1,   113,   114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79,    80,    81,    -1,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    -1,
      -1,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,
     113,   114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    80,    81,    -1,    83,
      84,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,    56,    -1,    58,    59,    -1,    -1,    -1,    -1,   113,
     114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    -1,    -1,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
      56,    -1,    58,    59,    -1,    -1,    -1,    -1,   113,   114,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,    84,    85,
      86,    87,    88,    89,    90,    91,    -1,    -1,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   104,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   113,   114
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   117,   118,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    30,    31,    33,    34,    35,    38,
      39,    40,    41,    42,    43,    45,    47,    48,    49,    50,
      51,    52,    53,    54,    56,    58,    60,    82,    86,    92,
      93,    97,   103,   105,   106,   107,   108,   109,   110,   112,
     113,   114,   115,   119,   120,   122,   123,   124,   126,   127,
     129,   130,   131,   134,   136,   137,   145,   146,   147,   155,
     156,   157,   164,   166,   179,   181,   190,   192,   199,   200,
     201,   203,   205,   213,   214,   215,   217,   219,   222,   236,
     241,   246,   250,   251,   252,   253,   254,   255,   256,   258,
     260,   264,   266,   267,   268,   269,   270,     3,    63,     3,
       1,     6,   125,   252,     1,    45,   254,     1,     3,     1,
       3,    14,     1,   254,     1,   252,   273,     1,   254,     1,
       1,   254,     1,     3,    44,     1,   254,     1,     6,     1,
       6,     1,     3,   254,   247,     1,     6,     7,     1,   254,
     256,     1,     6,     1,     3,     6,   216,     1,     6,    32,
     218,     1,     6,   220,   221,     1,     6,   259,   265,     1,
     254,    57,   254,   271,     1,     3,    44,     6,   254,    44,
      57,    62,   254,   270,   274,   261,   254,     1,     3,   254,
     270,   254,   254,   254,     1,     3,   270,   254,   254,   254,
     254,   254,   254,     4,     6,    26,   254,   254,     4,   252,
     128,    31,    33,    45,   123,   135,   123,   165,   180,   191,
      46,   208,   211,   212,   123,     1,    56,     3,    56,    58,
      59,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    79,    80,    81,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   113,   114,   255,    75,
      78,   254,     3,     3,    78,    75,     3,    44,     3,    44,
       3,     3,     3,     3,    44,     3,    44,     3,    75,    78,
      91,     3,     3,     3,     3,     3,     3,   123,     3,     3,
       3,   223,     3,   242,     3,     3,     6,   248,   249,     3,
       3,     3,     3,     3,     3,    75,     3,     3,    78,     3,
       1,     6,     7,     1,     3,    32,    78,     3,    75,     3,
      78,     3,     1,    56,   262,   262,     3,     3,     1,    57,
      78,   272,     3,   132,   123,   237,    55,    57,   254,    57,
      44,    62,     1,    57,     1,    57,    78,     1,     6,   206,
     207,   263,     3,     3,     3,     3,   144,   144,   167,   182,
     144,     1,     3,    44,   144,   209,   210,     3,     1,   206,
      55,   270,   101,   254,     6,   254,   254,   254,   254,   254,
     254,   254,   254,   254,   254,   254,   254,     1,   254,   254,
     254,   254,   254,   254,   254,   254,   254,     6,   254,   254,
     254,   254,   254,   254,   254,   254,   254,   254,   254,   254,
     254,   254,   254,   252,   254,   151,   254,   252,     1,   254,
       1,    56,   224,   225,     1,    32,   227,   243,     3,    78,
       1,   251,     6,     3,     3,    76,     3,    76,     3,     6,
       7,     6,     4,     6,     7,    97,   121,   221,     3,   206,
     208,   208,   254,   144,     3,    57,    57,   254,   254,    57,
     254,    62,     1,    62,    78,   208,     9,   123,    16,    17,
     138,   140,   141,   143,     1,     3,    23,    24,   168,   173,
     175,     1,     3,    23,   173,   183,    29,   193,   194,   195,
     196,     3,    44,     9,   144,   123,   204,     1,    55,     1,
      55,   254,    57,    78,     1,    44,     3,    78,    75,   149,
      77,     3,   148,     3,   206,   231,   227,     3,     6,   228,
     229,     3,   244,     1,   249,     3,     3,     6,     6,     3,
      76,    91,     3,    76,    91,     4,     1,    55,   144,   144,
       9,   238,    44,    57,    62,   207,   144,     3,     1,     3,
       1,   254,     9,   139,   142,     3,     1,     6,     7,     8,
     121,   177,   178,     1,     9,   174,     3,     1,     4,     6,
     188,   189,     9,     1,     3,     4,     6,    91,   197,   198,
       9,   195,   144,     3,     9,    55,   202,     3,    44,   257,
      57,   270,     1,   254,   270,   254,     3,    44,   150,     1,
     254,   150,     1,    55,     3,     6,     9,    37,    45,    46,
     200,   232,   233,   235,     3,    56,   230,    78,     3,     9,
     200,   233,   235,   245,     3,     3,     6,     6,     6,     6,
       3,     9,     9,   133,   254,     3,     6,     9,   239,   254,
     254,    61,     3,     3,     3,     3,   144,   144,     3,    44,
      77,     3,    44,    78,     3,     3,    44,   176,     3,    44,
       3,    44,    78,     3,     3,   252,     3,    78,    91,     3,
       3,    44,    55,    55,     3,    19,    20,    21,   123,   153,
     154,   158,   160,   162,   123,     1,    78,   152,   226,    75,
       3,     3,     6,    55,   270,   229,     3,     3,     3,     3,
       3,     3,     3,    75,    78,   240,     3,    57,   138,   171,
     172,   121,   169,   170,   178,   144,   123,   186,   187,   184,
     185,   189,     3,   198,   252,     3,     1,     3,    44,     1,
       3,    44,     1,     3,    44,     9,   153,     1,   254,    55,
     254,   234,    75,    55,   251,   144,   123,   144,   123,   144,
     123,   144,   123,     3,     3,   159,   123,     3,   161,   123,
       3,   163,   123,     3,     3,   208,   254,   240,   144,   144,
     144,   144,     3,     9,     9,     9,     9,     3,     3,     3,
       3
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
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:
#line 215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 230 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:
#line 235 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:
#line 246 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:
#line 251 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:
#line 257 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:
#line 263 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 25:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 29:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) ); }
    break;

  case 30:
#line 281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 31:
#line 287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 49:
#line 321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 50:
#line 327 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 51:
#line 336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 52:
#line 338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 53:
#line 342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 54:
#line 349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 55:
#line 356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 56:
#line 364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 57:
#line 365 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 58:
#line 369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 370 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 60:
#line 374 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 61:
#line 381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 62:
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 63:
#line 395 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
   }
    break;

  case 64:
#line 401 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 65:
#line 402 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 66:
#line 406 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 67:
#line 414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 68:
#line 421 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 69:
#line 431 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 70:
#line 432 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 71:
#line 436 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:
#line 437 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 75:
#line 444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 78:
#line 454 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 79:
#line 459 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 81:
#line 471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 82:
#line 472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 84:
#line 477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 85:
#line 484 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 86:
#line 493 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 87:
#line 501 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 88:
#line 511 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 89:
#line 520 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 90:
#line 529 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (4)].fal_adecl);
         if ( decl->front() == decl->back() ) {
            f = new Falcon::StmtForin( LINE, (Falcon::Value *) decl->front(), (yyvsp[(4) - (4)].fal_val) );
            decl->deletor(0);
            delete decl;
         }
         else
            f = new Falcon::StmtForin( LINE, new Falcon::Value(decl), (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 91:
#line 546 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 92:
#line 555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         f = new Falcon::StmtForin( LINE, (yyvsp[(2) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 93:
#line 566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 94:
#line 576 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 95:
#line 581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 96:
#line 589 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 98:
#line 602 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 99:
#line 608 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 100:
#line 612 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 101:
#line 618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 102:
#line 619 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 103:
#line 620 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 106:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 110:
#line 643 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 111:
#line 656 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 112:
#line 664 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 113:
#line 668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 114:
#line 674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 115:
#line 680 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 116:
#line 687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 117:
#line 692 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 118:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 119:
#line 710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 120:
#line 722 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 121:
#line 724 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 122:
#line 733 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 123:
#line 737 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 124:
#line 749 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 125:
#line 750 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 126:
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 127:
#line 763 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 128:
#line 777 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 129:
#line 779 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 130:
#line 788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 131:
#line 792 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 132:
#line 800 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 133:
#line 809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 134:
#line 811 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 137:
#line 820 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 139:
#line 826 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 141:
#line 836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 142:
#line 844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 143:
#line 848 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 145:
#line 860 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 146:
#line 870 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 148:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 152:
#line 893 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 154:
#line 897 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 157:
#line 909 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 158:
#line 918 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 159:
#line 930 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 160:
#line 941 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 161:
#line 952 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 162:
#line 972 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 163:
#line 980 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 164:
#line 989 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 165:
#line 991 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 168:
#line 1000 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 170:
#line 1006 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 172:
#line 1016 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 173:
#line 1025 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 174:
#line 1029 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 176:
#line 1041 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 177:
#line 1051 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 181:
#line 1065 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 182:
#line 1077 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 183:
#line 1097 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 184:
#line 1104 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 185:
#line 1114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 187:
#line 1123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 193:
#line 1143 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 194:
#line 1161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 195:
#line 1181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 196:
#line 1191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 197:
#line 1202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 200:
#line 1215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 201:
#line 1227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 202:
#line 1249 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 203:
#line 1250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 204:
#line 1262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 205:
#line 1268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 207:
#line 1277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 208:
#line 1278 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 209:
#line 1281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 211:
#line 1286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 212:
#line 1287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 213:
#line 1294 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
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

  case 217:
#line 1355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 219:
#line 1372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 220:
#line 1378 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 221:
#line 1383 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 222:
#line 1389 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 224:
#line 1398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 226:
#line 1403 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 227:
#line 1413 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 228:
#line 1416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 229:
#line 1425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 230:
#line 1435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 231:
#line 1440 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 232:
#line 1452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 233:
#line 1461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 234:
#line 1468 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 235:
#line 1476 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 236:
#line 1481 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 237:
#line 1489 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 238:
#line 1494 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 239:
#line 1499 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:
#line 1504 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );

         (yyval.fal_stat) = 0;
      }
    break;

  case 241:
#line 1524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // destroy the list to avoid leak
         Falcon::ListElement *li = (yyvsp[(2) - (7)].fal_genericList)->begin();
         int counter = 0;
         while( li != 0 ) {
            Falcon::String *symName = (Falcon::String *) li->data();
            if ( counter == 0 )
               COMPILER->importAlias( symName, (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
            delete symName;
            li = li->next();
            counter++;
         }
         delete (yyvsp[(2) - (7)].fal_genericList);

         if ( counter != 1 )
            COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:
#line 1543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 243:
#line 1548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 244:
#line 1553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:
#line 1558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 246:
#line 1572 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:
#line 1577 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 248:
#line 1582 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:
#line 1587 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:
#line 1592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:
#line 1600 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 252:
#line 1606 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 253:
#line 1618 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 254:
#line 1623 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 257:
#line 1636 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 258:
#line 1640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 259:
#line 1644 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 260:
#line 1657 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 261:
#line 1689 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( cd->inheritance().size() != 0 )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 263:
#line 1721 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 266:
#line 1729 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 267:
#line 1730 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 272:
#line 1747 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>( COMPILER->getContext() );
         // creates or find the symbol.
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol((yyvsp[(1) - (2)].stringp));
         Falcon::ClassDef *clsdef = cls->symbol()->getClassDef();
         Falcon::InheritDef *idef = new Falcon::InheritDef(sym);

         if ( clsdef->addInheritance( idef ) )
         {
            cls->addInitExpression(
               new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_inherit,
                     new Falcon::Value( sym ), (yyvsp[(2) - (2)].fal_val) ) ) );
         }
         else {
            COMPILER->raiseError(Falcon::e_prop_adef );
            delete idef;
            delete (yyvsp[(2) - (2)].fal_val);
         }
      }
    break;

  case 273:
#line 1770 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 274:
#line 1771 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 275:
#line 1773 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 279:
#line 1786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 280:
#line 1789 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 282:
#line 1811 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 283:
#line 1835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 284:
#line 1844 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 285:
#line 1866 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 286:
#line 1899 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 287:
#line 1933 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 291:
#line 1950 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 292:
#line 1955 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 295:
#line 1970 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
         cls->singleton( sym );

         // prepare the statement allocation context
         COMPILER->pushContext( cls );

         //Statements here goes in the auto constructor.
         //COMPILER->pushContextSet( &cls->autoCtor() );
         COMPILER->pushFunction( def );
      }
    break;

  case 296:
#line 2010 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         // if the class has no constructor, create one in case of inheritance.
         if( cls->ctorFunction() == 0  )
         {
            Falcon::ClassDef *cd = cls->symbol()->getClassDef();
            if ( !cd->inheritance().empty() )
            {
               COMPILER->buildCtorFor( cls );
               // COMPILER->addStatement( func ); should be done in buildCtorFor
               // cls->ctorFunction( func ); idem
            }
         }

         COMPILER->popContext();
         //COMPILER->popContextSet();
         COMPILER->popFunction();
      }
    break;

  case 298:
#line 2038 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 302:
#line 2050 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 303:
#line 2053 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 305:
#line 2081 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 306:
#line 2086 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 309:
#line 2101 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 310:
#line 2108 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 311:
#line 2123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 312:
#line 2124 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 313:
#line 2125 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 314:
#line 2135 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 315:
#line 2136 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 316:
#line 2137 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 317:
#line 2138 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 318:
#line 2139 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 319:
#line 2140 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 320:
#line 2145 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 322:
#line 2163 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 323:
#line 2164 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtFunction *sfunc = COMPILER->getFunctionContext();
      if ( sfunc == 0 ) {
         COMPILER->raiseError(Falcon::e_fself_outside, COMPILER->tempLine() );
         (yyval.fal_val) = new Falcon::Value();
      }
      else
      {
         (yyval.fal_val) = new Falcon::Value( sfunc->symbol() );
      }
   }
    break;

  case 326:
#line 2187 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 327:
#line 2188 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 328:
#line 2189 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 329:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 330:
#line 2191 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 331:
#line 2192 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 332:
#line 2193 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 333:
#line 2194 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 334:
#line 2195 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 335:
#line 2196 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 336:
#line 2197 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 337:
#line 2198 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 338:
#line 2199 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 339:
#line 2200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 340:
#line 2201 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 341:
#line 2202 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 342:
#line 2203 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 343:
#line 2204 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 344:
#line 2205 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 345:
#line 2206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 346:
#line 2207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 347:
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 348:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 349:
#line 2210 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 350:
#line 2211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 351:
#line 2212 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 352:
#line 2213 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 353:
#line 2214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 354:
#line 2215 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 355:
#line 2216 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 356:
#line 2217 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 357:
#line 2218 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 358:
#line 2219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 359:
#line 2220 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 360:
#line 2221 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 361:
#line 2222 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 362:
#line 2223 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 363:
#line 2224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 364:
#line 2225 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 365:
#line 2226 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 366:
#line 2227 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 367:
#line 2228 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 374:
#line 2236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 375:
#line 2241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 376:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 377:
#line 2250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 378:
#line 2256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 381:
#line 2268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 382:
#line 2273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 383:
#line 2280 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 384:
#line 2281 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 385:
#line 2282 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 386:
#line 2283 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 387:
#line 2284 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 388:
#line 2285 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 389:
#line 2286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 390:
#line 2287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 391:
#line 2288 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 392:
#line 2289 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 393:
#line 2290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 394:
#line 2291 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 395:
#line 2296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 396:
#line 2299 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 397:
#line 2302 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 398:
#line 2305 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 399:
#line 2308 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 400:
#line 2315 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 401:
#line 2321 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 402:
#line 2325 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 403:
#line 2326 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 404:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
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

  case 405:
#line 2369 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 406:
#line 2376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
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

  case 407:
#line 2410 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StatementList *stmt = COMPILER->getContextSet();
         if( stmt->size() == 1 && stmt->back()->type() == Falcon::Statement::t_autoexp )
         {
            // wrap it around a return, so A is not nilled.
            Falcon::StmtAutoexpr *ae = static_cast<Falcon::StmtAutoexpr *>( stmt->pop_back() );
            stmt->push_back( new Falcon::StmtReturn( 1, ae->value()->clone() ) );

            // we don't need the expression anymore.
            delete ae;
         }

         (yyval.fal_val) = COMPILER->closeClosure();
      }
    break;

  case 409:
#line 2429 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 410:
#line 2433 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 412:
#line 2441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 413:
#line 2445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 414:
#line 2452 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::FuncDef *def = new Falcon::FuncDef( 0, 0 );
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

  case 415:
#line 2485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 416:
#line 2500 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 417:
#line 2505 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 418:
#line 2512 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 419:
#line 2519 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 420:
#line 2528 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 421:
#line 2530 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 422:
#line 2534 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 423:
#line 2541 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 424:
#line 2543 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 425:
#line 2547 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 426:
#line 2555 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 427:
#line 2556 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 428:
#line 2558 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 429:
#line 2565 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 430:
#line 2566 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 431:
#line 2570 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 432:
#line 2571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 435:
#line 2578 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 436:
#line 2584 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 437:
#line 2591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 438:
#line 2592 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6457 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2596 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */



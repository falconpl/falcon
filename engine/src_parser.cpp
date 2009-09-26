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
     INIT = 291,
     LOAD = 292,
     LAUNCH = 293,
     CONST_KW = 294,
     EXPORT = 295,
     IMPORT = 296,
     DIRECTIVE = 297,
     COLON = 298,
     FUNCDECL = 299,
     STATIC = 300,
     INNERFUNC = 301,
     FORDOT = 302,
     LISTPAR = 303,
     LOOP = 304,
     ENUM = 305,
     TRUE_TOKEN = 306,
     FALSE_TOKEN = 307,
     OUTER_STRING = 308,
     CLOSEPAR = 309,
     OPENPAR = 310,
     CLOSESQUARE = 311,
     OPENSQUARE = 312,
     DOT = 313,
     OPEN_GRAPH = 314,
     CLOSE_GRAPH = 315,
     ARROW = 316,
     VBAR = 317,
     ASSIGN_POW = 318,
     ASSIGN_SHL = 319,
     ASSIGN_SHR = 320,
     ASSIGN_BXOR = 321,
     ASSIGN_BOR = 322,
     ASSIGN_BAND = 323,
     ASSIGN_MOD = 324,
     ASSIGN_DIV = 325,
     ASSIGN_MUL = 326,
     ASSIGN_SUB = 327,
     ASSIGN_ADD = 328,
     OP_EQ = 329,
     OP_AS = 330,
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
     DIESIS = 346,
     ATSIGN = 347,
     CAP_CAP = 348,
     VBAR_VBAR = 349,
     AMPER_AMPER = 350,
     MINUS = 351,
     PLUS = 352,
     PERCENT = 353,
     SLASH = 354,
     STAR = 355,
     POW = 356,
     SHR = 357,
     SHL = 358,
     CAP_XOROOB = 359,
     CAP_ISOOB = 360,
     CAP_DEOOB = 361,
     CAP_OOB = 362,
     CAP_EVAL = 363,
     TILDE = 364,
     NEG = 365,
     AMPER = 366,
     DECREMENT = 367,
     INCREMENT = 368,
     DOLLAR = 369
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
#define INIT 291
#define LOAD 292
#define LAUNCH 293
#define CONST_KW 294
#define EXPORT 295
#define IMPORT 296
#define DIRECTIVE 297
#define COLON 298
#define FUNCDECL 299
#define STATIC 300
#define INNERFUNC 301
#define FORDOT 302
#define LISTPAR 303
#define LOOP 304
#define ENUM 305
#define TRUE_TOKEN 306
#define FALSE_TOKEN 307
#define OUTER_STRING 308
#define CLOSEPAR 309
#define OPENPAR 310
#define CLOSESQUARE 311
#define OPENSQUARE 312
#define DOT 313
#define OPEN_GRAPH 314
#define CLOSE_GRAPH 315
#define ARROW 316
#define VBAR 317
#define ASSIGN_POW 318
#define ASSIGN_SHL 319
#define ASSIGN_SHR 320
#define ASSIGN_BXOR 321
#define ASSIGN_BOR 322
#define ASSIGN_BAND 323
#define ASSIGN_MOD 324
#define ASSIGN_DIV 325
#define ASSIGN_MUL 326
#define ASSIGN_SUB 327
#define ASSIGN_ADD 328
#define OP_EQ 329
#define OP_AS 330
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
#define DIESIS 346
#define ATSIGN 347
#define CAP_CAP 348
#define VBAR_VBAR 349
#define AMPER_AMPER 350
#define MINUS 351
#define PLUS 352
#define PERCENT 353
#define SLASH 354
#define STAR 355
#define POW 356
#define SHR 357
#define SHL 358
#define CAP_XOROOB 359
#define CAP_ISOOB 360
#define CAP_DEOOB 361
#define CAP_OOB 362
#define CAP_EVAL 363
#define TILDE 364
#define NEG 365
#define AMPER 366
#define DECREMENT 367
#define INCREMENT 368
#define DOLLAR 369




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
#define  CTX_LINE  ( COMPILER->lexer()->contextStart() )
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
#line 389 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 402 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"

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
#define YYLAST   6633

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  115
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  162
/* YYNRULES -- Number of rules.  */
#define YYNRULES  458
/* YYNRULES -- Number of states.  */
#define YYNSTATES  835

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   369

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
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    14,    18,    20,
      22,    24,    26,    28,    30,    32,    34,    36,    38,    40,
      43,    47,    51,    55,    57,    59,    61,    65,    69,    73,
      76,    79,    84,    91,    93,    95,    97,    99,   101,   103,
     105,   107,   109,   111,   113,   115,   117,   119,   121,   123,
     125,   127,   131,   137,   141,   145,   146,   152,   155,   159,
     163,   167,   171,   172,   180,   184,   188,   189,   191,   192,
     199,   202,   206,   210,   214,   218,   219,   221,   222,   226,
     229,   233,   234,   239,   243,   247,   248,   251,   254,   258,
     261,   265,   269,   270,   277,   278,   285,   291,   295,   298,
     303,   308,   313,   317,   318,   321,   324,   325,   328,   330,
     332,   334,   336,   340,   344,   348,   351,   355,   358,   362,
     366,   368,   369,   376,   380,   384,   385,   392,   396,   400,
     401,   408,   412,   416,   417,   424,   428,   432,   433,   436,
     440,   442,   443,   449,   450,   456,   457,   463,   464,   470,
     471,   472,   476,   477,   479,   482,   485,   488,   490,   494,
     496,   498,   500,   504,   506,   507,   514,   518,   522,   523,
     526,   530,   532,   533,   539,   540,   546,   547,   553,   554,
     560,   562,   566,   567,   569,   571,   575,   576,   583,   586,
     590,   591,   593,   595,   598,   601,   604,   609,   613,   619,
     623,   625,   629,   631,   633,   637,   641,   647,   650,   656,
     657,   665,   669,   675,   676,   683,   686,   687,   689,   693,
     695,   696,   697,   703,   704,   708,   711,   715,   718,   722,
     726,   730,   736,   742,   746,   749,   753,   757,   759,   763,
     767,   773,   779,   787,   795,   803,   811,   816,   821,   826,
     831,   838,   845,   849,   854,   859,   861,   865,   869,   873,
     875,   879,   883,   887,   891,   892,   900,   904,   907,   908,
     912,   913,   919,   920,   923,   925,   929,   932,   933,   936,
     940,   941,   944,   946,   948,   950,   952,   954,   955,   963,
     969,   974,   975,   983,   984,   987,   989,   994,   996,   999,
    1001,  1003,  1004,  1012,  1015,  1018,  1019,  1022,  1024,  1026,
    1028,  1030,  1032,  1033,  1038,  1040,  1042,  1045,  1049,  1053,
    1055,  1058,  1062,  1066,  1068,  1070,  1072,  1074,  1076,  1078,
    1080,  1082,  1084,  1086,  1088,  1090,  1092,  1094,  1096,  1098,
    1099,  1101,  1103,  1105,  1108,  1111,  1114,  1118,  1122,  1126,
    1129,  1133,  1138,  1143,  1148,  1153,  1158,  1163,  1168,  1173,
    1178,  1183,  1188,  1191,  1195,  1198,  1201,  1204,  1207,  1211,
    1215,  1219,  1223,  1227,  1231,  1235,  1238,  1242,  1246,  1250,
    1253,  1256,  1259,  1262,  1265,  1268,  1271,  1274,  1277,  1279,
    1281,  1283,  1285,  1287,  1289,  1292,  1294,  1299,  1305,  1309,
    1311,  1313,  1317,  1323,  1327,  1331,  1335,  1339,  1343,  1347,
    1351,  1355,  1359,  1363,  1367,  1371,  1375,  1380,  1385,  1391,
    1399,  1404,  1408,  1409,  1416,  1417,  1424,  1425,  1432,  1437,
    1441,  1444,  1447,  1450,  1453,  1454,  1461,  1467,  1473,  1478,
    1482,  1485,  1489,  1493,  1496,  1500,  1504,  1508,  1512,  1517,
    1519,  1523,  1525,  1529,  1530,  1532,  1534,  1538,  1542
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     116,     0,    -1,   117,    -1,    -1,   117,   118,    -1,   119,
      -1,     9,     3,    -1,    23,     1,     3,    -1,   121,    -1,
     219,    -1,   199,    -1,   222,    -1,   241,    -1,   236,    -1,
     122,    -1,   213,    -1,   214,    -1,   216,    -1,     4,    -1,
      96,     4,    -1,    37,     6,     3,    -1,    37,     7,     3,
      -1,    37,     1,     3,    -1,   123,    -1,   217,    -1,     3,
      -1,    44,     1,     3,    -1,    33,     1,     3,    -1,    31,
       1,     3,    -1,     1,     3,    -1,   256,     3,    -1,   272,
      74,   256,     3,    -1,   272,    74,   256,    77,   272,     3,
      -1,   125,    -1,   126,    -1,   130,    -1,   146,    -1,   163,
      -1,   178,    -1,   133,    -1,   144,    -1,   145,    -1,   189,
      -1,   198,    -1,   250,    -1,   246,    -1,   212,    -1,   154,
      -1,   155,    -1,   156,    -1,   253,    -1,   253,    74,   256,
      -1,   124,    77,   253,    74,   256,    -1,    10,   124,     3,
      -1,    10,     1,     3,    -1,    -1,   128,   127,   143,     9,
       3,    -1,   129,   122,    -1,    11,   256,     3,    -1,    11,
       1,     3,    -1,    11,   256,    43,    -1,    11,     1,    43,
      -1,    -1,    49,     3,   131,   143,     9,   132,     3,    -1,
      49,    43,   122,    -1,    49,     1,     3,    -1,    -1,   256,
      -1,    -1,   135,   134,   143,   137,     9,     3,    -1,   136,
     122,    -1,    15,   256,     3,    -1,    15,     1,     3,    -1,
      15,   256,    43,    -1,    15,     1,    43,    -1,    -1,   140,
      -1,    -1,   139,   138,   143,    -1,    16,     3,    -1,    16,
       1,     3,    -1,    -1,   142,   141,   143,   137,    -1,    17,
     256,     3,    -1,    17,     1,     3,    -1,    -1,   143,   122,
      -1,    12,     3,    -1,    12,     1,     3,    -1,    13,     3,
      -1,    13,    14,     3,    -1,    13,     1,     3,    -1,    -1,
      18,   275,    90,   256,   147,   149,    -1,    -1,    18,   253,
      74,   150,   148,   149,    -1,    18,   275,    90,     1,     3,
      -1,    18,     1,     3,    -1,    43,   122,    -1,     3,   152,
       9,     3,    -1,   256,    76,   256,   151,    -1,   256,    76,
     256,     1,    -1,   256,    76,     1,    -1,    -1,    77,   256,
      -1,    77,     1,    -1,    -1,   153,   152,    -1,   122,    -1,
     157,    -1,   159,    -1,   161,    -1,    47,   256,     3,    -1,
      47,     1,     3,    -1,   102,   272,     3,    -1,   102,     3,
      -1,    85,   272,     3,    -1,    85,     3,    -1,   102,     1,
       3,    -1,    85,     1,     3,    -1,    53,    -1,    -1,    19,
       3,   158,   143,     9,     3,    -1,    19,    43,   122,    -1,
      19,     1,     3,    -1,    -1,    20,     3,   160,   143,     9,
       3,    -1,    20,    43,   122,    -1,    20,     1,     3,    -1,
      -1,    21,     3,   162,   143,     9,     3,    -1,    21,    43,
     122,    -1,    21,     1,     3,    -1,    -1,   165,   164,   166,
     172,     9,     3,    -1,    22,   256,     3,    -1,    22,     1,
       3,    -1,    -1,   166,   167,    -1,   166,     1,     3,    -1,
       3,    -1,    -1,    23,   176,     3,   168,   143,    -1,    -1,
      23,   176,    43,   169,   122,    -1,    -1,    23,     1,     3,
     170,   143,    -1,    -1,    23,     1,    43,   171,   122,    -1,
      -1,    -1,   174,   173,   175,    -1,    -1,    24,    -1,    24,
       1,    -1,     3,   143,    -1,    43,   122,    -1,   177,    -1,
     176,    77,   177,    -1,     8,    -1,   120,    -1,     7,    -1,
     120,    76,   120,    -1,     6,    -1,    -1,   180,   179,   181,
     172,     9,     3,    -1,    25,   256,     3,    -1,    25,     1,
       3,    -1,    -1,   181,   182,    -1,   181,     1,     3,    -1,
       3,    -1,    -1,    23,   187,     3,   183,   143,    -1,    -1,
      23,   187,    43,   184,   122,    -1,    -1,    23,     1,     3,
     185,   143,    -1,    -1,    23,     1,    43,   186,   122,    -1,
     188,    -1,   187,    77,   188,    -1,    -1,     4,    -1,     6,
      -1,    28,    43,   122,    -1,    -1,   191,   190,   143,   192,
       9,     3,    -1,    28,     3,    -1,    28,     1,     3,    -1,
      -1,   193,    -1,   194,    -1,   193,   194,    -1,   195,   143,
      -1,    29,     3,    -1,    29,    90,   253,     3,    -1,    29,
     196,     3,    -1,    29,   196,    90,   253,     3,    -1,    29,
       1,     3,    -1,   197,    -1,   196,    77,   197,    -1,     4,
      -1,     6,    -1,    30,   256,     3,    -1,    30,     1,     3,
      -1,   200,   207,   143,     9,     3,    -1,   202,   122,    -1,
     204,    55,   205,    54,     3,    -1,    -1,   204,    55,   205,
       1,   201,    54,     3,    -1,   204,     1,     3,    -1,   204,
      55,   205,    54,    43,    -1,    -1,   204,    55,     1,   203,
      54,    43,    -1,    44,     6,    -1,    -1,   206,    -1,   205,
      77,   206,    -1,     6,    -1,    -1,    -1,   210,   208,   143,
       9,     3,    -1,    -1,   211,   209,   122,    -1,    45,     3,
      -1,    45,     1,     3,    -1,    45,    43,    -1,    45,     1,
      43,    -1,    38,   258,     3,    -1,    38,     1,     3,    -1,
      39,     6,    74,   252,     3,    -1,    39,     6,    74,     1,
       3,    -1,    39,     1,     3,    -1,    40,     3,    -1,    40,
     215,     3,    -1,    40,     1,     3,    -1,     6,    -1,   215,
      77,     6,    -1,    41,   218,     3,    -1,    41,   218,    32,
       6,     3,    -1,    41,   218,    32,     7,     3,    -1,    41,
     218,    32,     6,    75,     6,     3,    -1,    41,   218,    32,
       7,    75,     6,     3,    -1,    41,   218,    32,     6,    90,
       6,     3,    -1,    41,   218,    32,     7,    90,     6,     3,
      -1,    41,     6,     1,     3,    -1,    41,   218,     1,     3,
      -1,    41,    32,     6,     3,    -1,    41,    32,     7,     3,
      -1,    41,    32,     6,    75,     6,     3,    -1,    41,    32,
       7,    75,     6,     3,    -1,    41,     1,     3,    -1,     6,
      43,   252,     3,    -1,     6,    43,     1,     3,    -1,     6,
      -1,   218,    77,     6,    -1,    42,   220,     3,    -1,    42,
       1,     3,    -1,   221,    -1,   220,    77,   221,    -1,     6,
      74,     6,    -1,     6,    74,     7,    -1,     6,    74,   120,
      -1,    -1,    31,     6,   223,   224,   231,     9,     3,    -1,
     225,   227,     3,    -1,     1,     3,    -1,    -1,    55,   205,
      54,    -1,    -1,    55,   205,     1,   226,    54,    -1,    -1,
      32,   228,    -1,   229,    -1,   228,    77,   229,    -1,     6,
     230,    -1,    -1,    55,    54,    -1,    55,   272,    54,    -1,
      -1,   231,   232,    -1,     3,    -1,   199,    -1,   235,    -1,
     233,    -1,   217,    -1,    -1,    36,     3,   234,   207,   143,
       9,     3,    -1,    45,     6,    74,   252,     3,    -1,     6,
      74,   256,     3,    -1,    -1,    50,     6,   237,     3,   238,
       9,     3,    -1,    -1,   238,   239,    -1,     3,    -1,     6,
      74,   252,   240,    -1,   217,    -1,     6,   240,    -1,     3,
      -1,    77,    -1,    -1,    33,     6,   242,   243,   244,     9,
       3,    -1,   227,     3,    -1,     1,     3,    -1,    -1,   244,
     245,    -1,     3,    -1,   199,    -1,   235,    -1,   233,    -1,
     217,    -1,    -1,    35,   247,   248,     3,    -1,   249,    -1,
       1,    -1,   249,     1,    -1,   248,    77,   249,    -1,   248,
      77,     1,    -1,     6,    -1,    34,     3,    -1,    34,   256,
       3,    -1,    34,     1,     3,    -1,     8,    -1,    51,    -1,
      52,    -1,     4,    -1,     5,    -1,     7,    -1,     8,    -1,
      51,    -1,    52,    -1,   120,    -1,     5,    -1,     7,    -1,
       6,    -1,   253,    -1,    26,    -1,    27,    -1,    -1,     3,
      -1,   251,    -1,   254,    -1,   111,     6,    -1,   111,     4,
      -1,   111,    26,    -1,   111,    58,     6,    -1,   111,    58,
       4,    -1,   111,    58,    26,    -1,    96,   256,    -1,     6,
      62,   256,    -1,   256,    97,   255,   256,    -1,   256,    96,
     255,   256,    -1,   256,   100,   255,   256,    -1,   256,    99,
     255,   256,    -1,   256,    98,   255,   256,    -1,   256,   101,
     255,   256,    -1,   256,    95,   255,   256,    -1,   256,    94,
     255,   256,    -1,   256,    93,   255,   256,    -1,   256,   103,
     255,   256,    -1,   256,   102,   255,   256,    -1,   109,   256,
      -1,   256,    86,   256,    -1,   256,   113,    -1,   113,   256,
      -1,   256,   112,    -1,   112,   256,    -1,   256,    87,   256,
      -1,   256,    85,   256,    -1,   256,    84,   256,    -1,   256,
      83,   256,    -1,   256,    82,   256,    -1,   256,    80,   256,
      -1,   256,    79,   256,    -1,    81,   256,    -1,   256,    90,
     256,    -1,   256,    89,   256,    -1,   256,    88,     6,    -1,
     114,   253,    -1,   114,     4,    -1,    92,   256,    -1,    91,
     256,    -1,   108,   256,    -1,   107,   256,    -1,   106,   256,
      -1,   105,   256,    -1,   104,   256,    -1,   260,    -1,   262,
      -1,   266,    -1,   258,    -1,   268,    -1,   270,    -1,   256,
     257,    -1,   269,    -1,   256,    57,   256,    56,    -1,   256,
      57,   100,   256,    56,    -1,   256,    58,     6,    -1,   271,
      -1,   257,    -1,   256,    74,   256,    -1,   256,    74,   256,
      77,   272,    -1,   256,    73,   256,    -1,   256,    72,   256,
      -1,   256,    71,   256,    -1,   256,    70,   256,    -1,   256,
      69,   256,    -1,   256,    63,   256,    -1,   256,    68,   256,
      -1,   256,    67,   256,    -1,   256,    66,   256,    -1,   256,
      64,   256,    -1,   256,    65,   256,    -1,    55,   256,    54,
      -1,    57,    43,    56,    -1,    57,   256,    43,    56,    -1,
      57,    43,   256,    56,    -1,    57,   256,    43,   256,    56,
      -1,    57,   256,    43,   256,    43,   256,    56,    -1,   256,
      55,   272,    54,    -1,   256,    55,    54,    -1,    -1,   256,
      55,   272,     1,   259,    54,    -1,    -1,    44,   261,   264,
     207,   143,     9,    -1,    -1,    59,   263,   265,   207,   143,
      60,    -1,    55,   205,    54,     3,    -1,    55,   205,     1,
      -1,     1,     3,    -1,   205,    61,    -1,   205,     1,    -1,
       1,    61,    -1,    -1,    46,   267,   264,   207,   143,     9,
      -1,   256,    78,   256,    43,   256,    -1,   256,    78,   256,
      43,     1,    -1,   256,    78,   256,     1,    -1,   256,    78,
       1,    -1,    57,    56,    -1,    57,   272,    56,    -1,    57,
     272,     1,    -1,    48,    56,    -1,    48,   273,    56,    -1,
      48,   273,     1,    -1,    57,    61,    56,    -1,    57,   276,
      56,    -1,    57,   276,     1,    56,    -1,   256,    -1,   272,
      77,   256,    -1,   256,    -1,   273,   274,   256,    -1,    -1,
      77,    -1,   253,    -1,   275,    77,   253,    -1,   256,    61,
     256,    -1,   276,    77,   256,    61,   256,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   196,   196,   199,   201,   205,   206,   207,   211,   212,
     213,   218,   223,   228,   233,   238,   239,   240,   244,   245,
     249,   255,   261,   268,   269,   270,   271,   272,   273,   274,
     279,   290,   296,   310,   311,   312,   313,   314,   315,   316,
     317,   318,   319,   320,   321,   322,   323,   324,   325,   326,
     330,   333,   339,   347,   349,   354,   354,   368,   376,   377,
     381,   382,   386,   386,   401,   407,   414,   415,   419,   419,
     434,   444,   445,   449,   450,   454,   456,   457,   457,   466,
     467,   472,   472,   484,   485,   488,   490,   496,   505,   513,
     523,   532,   542,   541,   562,   561,   584,   589,   597,   603,
     610,   616,   620,   627,   628,   629,   632,   634,   638,   645,
     646,   647,   651,   664,   672,   676,   682,   688,   695,   700,
     709,   719,   719,   733,   742,   746,   746,   759,   768,   772,
     772,   788,   797,   801,   801,   818,   819,   826,   828,   829,
     833,   835,   834,   845,   845,   857,   857,   869,   869,   885,
     888,   887,   900,   901,   902,   905,   906,   912,   913,   917,
     926,   938,   949,   960,   981,   981,   998,   999,  1006,  1008,
    1009,  1013,  1015,  1014,  1025,  1025,  1038,  1038,  1050,  1050,
    1068,  1069,  1072,  1073,  1085,  1106,  1113,  1112,  1131,  1132,
    1135,  1137,  1141,  1142,  1146,  1151,  1169,  1189,  1199,  1210,
    1218,  1219,  1223,  1235,  1258,  1259,  1266,  1276,  1285,  1286,
    1286,  1290,  1294,  1295,  1295,  1302,  1356,  1358,  1359,  1363,
    1378,  1381,  1380,  1392,  1391,  1406,  1407,  1411,  1412,  1421,
    1425,  1433,  1443,  1448,  1460,  1469,  1476,  1484,  1489,  1497,
    1502,  1507,  1512,  1532,  1551,  1556,  1561,  1566,  1580,  1585,
    1590,  1595,  1600,  1609,  1614,  1621,  1627,  1639,  1644,  1652,
    1653,  1657,  1661,  1665,  1679,  1678,  1741,  1744,  1750,  1752,
    1753,  1753,  1759,  1761,  1765,  1766,  1770,  1794,  1795,  1796,
    1803,  1805,  1809,  1810,  1813,  1831,  1832,  1836,  1836,  1870,
    1892,  1926,  1925,  1969,  1971,  1975,  1976,  1980,  1981,  1988,
    1988,  1997,  1996,  2063,  2064,  2070,  2072,  2076,  2077,  2080,
    2099,  2100,  2109,  2108,  2126,  2127,  2132,  2137,  2138,  2145,
    2161,  2162,  2163,  2171,  2172,  2173,  2174,  2175,  2176,  2180,
    2181,  2182,  2183,  2184,  2185,  2189,  2207,  2208,  2209,  2229,
    2231,  2235,  2236,  2237,  2238,  2239,  2240,  2241,  2242,  2243,
    2244,  2245,  2271,  2272,  2292,  2316,  2333,  2334,  2335,  2336,
    2337,  2338,  2339,  2340,  2341,  2342,  2343,  2344,  2345,  2346,
    2347,  2348,  2349,  2350,  2351,  2352,  2353,  2354,  2355,  2356,
    2357,  2358,  2359,  2360,  2361,  2362,  2363,  2364,  2365,  2366,
    2367,  2368,  2369,  2370,  2372,  2377,  2381,  2386,  2392,  2401,
    2402,  2404,  2409,  2416,  2417,  2418,  2419,  2420,  2421,  2422,
    2423,  2424,  2425,  2426,  2427,  2432,  2435,  2438,  2441,  2444,
    2450,  2456,  2461,  2461,  2471,  2470,  2514,  2513,  2565,  2566,
    2570,  2577,  2578,  2582,  2590,  2589,  2639,  2644,  2651,  2658,
    2668,  2669,  2673,  2681,  2682,  2686,  2695,  2696,  2697,  2705,
    2706,  2710,  2711,  2714,  2715,  2718,  2724,  2731,  2732
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
  "RAISE", "CLASS", "FROM", "OBJECT", "RETURN", "GLOBAL", "INIT", "LOAD",
  "LAUNCH", "CONST_KW", "EXPORT", "IMPORT", "DIRECTIVE", "COLON",
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
  "export_symbol_list", "import_statement", "attribute_statement",
  "import_symbol_list", "directive_statement", "directive_pair_list",
  "directive_pair", "class_decl", "@27", "class_def_inner",
  "class_param_list", "@28", "from_clause", "inherit_list",
  "inherit_token", "inherit_call", "class_statement_list",
  "class_statement", "init_decl", "@29", "property_decl", "enum_statement",
  "@30", "enum_statement_list", "enum_item_decl", "enum_item_terminator",
  "object_decl", "@31", "object_decl_inner", "object_statement_list",
  "object_statement", "global_statement", "@32", "global_symbol_list",
  "globalized_symbol", "return_statement", "const_atom_non_minus",
  "const_atom", "atomic_symbol", "var_atom", "OPT_EOL", "expression",
  "range_decl", "func_call", "@33", "nameless_func", "@34",
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
     365,   366,   367,   368,   369
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint16 yyr1[] =
{
       0,   115,   116,   117,   117,   118,   118,   118,   119,   119,
     119,   119,   119,   119,   119,   119,   119,   119,   120,   120,
     121,   121,   121,   122,   122,   122,   122,   122,   122,   122,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     123,   123,   123,   123,   123,   123,   123,   123,   123,   123,
     124,   124,   124,   125,   125,   127,   126,   126,   128,   128,
     129,   129,   131,   130,   130,   130,   132,   132,   134,   133,
     133,   135,   135,   136,   136,   137,   137,   138,   137,   139,
     139,   141,   140,   142,   142,   143,   143,   144,   144,   145,
     145,   145,   147,   146,   148,   146,   146,   146,   149,   149,
     150,   150,   150,   151,   151,   151,   152,   152,   153,   153,
     153,   153,   154,   154,   155,   155,   155,   155,   155,   155,
     156,   158,   157,   157,   157,   160,   159,   159,   159,   162,
     161,   161,   161,   164,   163,   165,   165,   166,   166,   166,
     167,   168,   167,   169,   167,   170,   167,   171,   167,   172,
     173,   172,   174,   174,   174,   175,   175,   176,   176,   177,
     177,   177,   177,   177,   179,   178,   180,   180,   181,   181,
     181,   182,   183,   182,   184,   182,   185,   182,   186,   182,
     187,   187,   188,   188,   188,   189,   190,   189,   191,   191,
     192,   192,   193,   193,   194,   195,   195,   195,   195,   195,
     196,   196,   197,   197,   198,   198,   199,   199,   200,   201,
     200,   200,   202,   203,   202,   204,   205,   205,   205,   206,
     207,   208,   207,   209,   207,   210,   210,   211,   211,   212,
     212,   213,   213,   213,   214,   214,   214,   215,   215,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   217,   217,   218,   218,   219,   219,   220,
     220,   221,   221,   221,   223,   222,   224,   224,   225,   225,
     226,   225,   227,   227,   228,   228,   229,   230,   230,   230,
     231,   231,   232,   232,   232,   232,   232,   234,   233,   235,
     235,   237,   236,   238,   238,   239,   239,   239,   239,   240,
     240,   242,   241,   243,   243,   244,   244,   245,   245,   245,
     245,   245,   247,   246,   248,   248,   248,   248,   248,   249,
     250,   250,   250,   251,   251,   251,   251,   251,   251,   252,
     252,   252,   252,   252,   252,   253,   254,   254,   254,   255,
     255,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,   257,   257,   257,   257,   257,
     258,   258,   259,   258,   261,   260,   263,   262,   264,   264,
     264,   265,   265,   265,   267,   266,   268,   268,   268,   268,
     269,   269,   269,   270,   270,   270,   271,   271,   271,   272,
     272,   273,   273,   274,   274,   275,   275,   276,   276
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     1,     2,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       3,     3,     3,     1,     1,     1,     3,     3,     3,     2,
       2,     4,     6,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     5,     3,     3,     0,     5,     2,     3,     3,
       3,     3,     0,     7,     3,     3,     0,     1,     0,     6,
       2,     3,     3,     3,     3,     0,     1,     0,     3,     2,
       3,     0,     4,     3,     3,     0,     2,     2,     3,     2,
       3,     3,     0,     6,     0,     6,     5,     3,     2,     4,
       4,     4,     3,     0,     2,     2,     0,     2,     1,     1,
       1,     1,     3,     3,     3,     2,     3,     2,     3,     3,
       1,     0,     6,     3,     3,     0,     6,     3,     3,     0,
       6,     3,     3,     0,     6,     3,     3,     0,     2,     3,
       1,     0,     5,     0,     5,     0,     5,     0,     5,     0,
       0,     3,     0,     1,     2,     2,     2,     1,     3,     1,
       1,     1,     3,     1,     0,     6,     3,     3,     0,     2,
       3,     1,     0,     5,     0,     5,     0,     5,     0,     5,
       1,     3,     0,     1,     1,     3,     0,     6,     2,     3,
       0,     1,     1,     2,     2,     2,     4,     3,     5,     3,
       1,     3,     1,     1,     3,     3,     5,     2,     5,     0,
       7,     3,     5,     0,     6,     2,     0,     1,     3,     1,
       0,     0,     5,     0,     3,     2,     3,     2,     3,     3,
       3,     5,     5,     3,     2,     3,     3,     1,     3,     3,
       5,     5,     7,     7,     7,     7,     4,     4,     4,     4,
       6,     6,     3,     4,     4,     1,     3,     3,     3,     1,
       3,     3,     3,     3,     0,     7,     3,     2,     0,     3,
       0,     5,     0,     2,     1,     3,     2,     0,     2,     3,
       0,     2,     1,     1,     1,     1,     1,     0,     7,     5,
       4,     0,     7,     0,     2,     1,     4,     1,     2,     1,
       1,     0,     7,     2,     2,     0,     2,     1,     1,     1,
       1,     1,     0,     4,     1,     1,     2,     3,     3,     1,
       2,     3,     3,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     1,     1,     2,     2,     2,     3,     3,     3,     2,
       3,     4,     4,     4,     4,     4,     4,     4,     4,     4,
       4,     4,     2,     3,     2,     2,     2,     2,     3,     3,
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
       3,     0,     0,     1,     0,    25,   326,   327,   335,   328,
     323,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   337,   338,     0,     0,     0,     0,     0,   312,     0,
       0,     0,     0,     0,     0,     0,   434,     0,     0,     0,
       0,   324,   325,   120,     0,     0,   426,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     5,     8,    14,    23,    33,    34,
      55,     0,    35,    39,    68,     0,    40,    41,    36,    47,
      48,    49,    37,   133,    38,   164,    42,   186,    43,    10,
     220,     0,     0,    46,    15,    16,    17,    24,     9,    11,
      13,    12,    45,    44,   341,   336,   342,   449,   400,   391,
     388,   389,   390,   392,   395,   393,   399,     0,    29,     0,
       0,     6,     0,   335,     0,    50,     0,   335,   424,     0,
       0,    87,     0,    89,     0,     0,     0,     0,   455,     0,
       0,     0,     0,     0,     0,     0,   188,     0,     0,     0,
       0,   264,     0,   301,     0,   320,     0,     0,     0,     0,
       0,     0,     0,   391,     0,     0,     0,   234,   237,     0,
       0,     0,     0,     0,     0,     0,     0,   259,     0,   215,
       0,     0,     0,     0,   443,   451,     0,     0,    62,     0,
     291,     0,     0,   440,     0,   449,     0,     0,     0,   375,
       0,   117,   449,     0,   382,   381,   349,     0,   115,     0,
     387,   386,   385,   384,   383,   362,   344,   343,   345,     0,
     367,   365,   380,   379,    85,     0,     0,     0,    57,    85,
      70,   137,   168,    85,     0,    85,   221,   223,   207,     0,
       0,    30,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   339,
     339,   339,   339,   339,   339,   339,   339,   339,   339,   339,
     366,   364,   394,     0,     0,     0,    18,   333,   334,   329,
     330,   331,     0,   332,     0,   350,    54,    53,     0,     0,
      59,    61,    58,    60,    88,    91,    90,    72,    74,    71,
      73,    97,     0,     0,     0,   136,   135,     7,   167,   166,
     189,   185,   205,   204,    28,     0,    27,     0,   322,   321,
     315,   319,     0,     0,    22,    20,    21,   230,   229,   233,
       0,   236,   235,     0,   252,     0,     0,     0,     0,   239,
       0,     0,   258,     0,   257,     0,    26,     0,   216,   220,
     220,   113,   112,   445,   444,   454,     0,    65,    85,    64,
       0,   414,   415,     0,   446,     0,     0,   442,   441,     0,
     447,     0,     0,   219,     0,   217,   220,   119,   116,   118,
     114,   347,   346,   348,     0,     0,     0,     0,     0,     0,
     225,   227,     0,    85,     0,   211,   213,     0,   421,     0,
       0,     0,   398,   408,   412,   413,   411,   410,   409,   407,
     406,   405,   404,   403,   401,   439,     0,   374,   373,   372,
     371,   370,   369,   363,   368,   378,   377,   376,   340,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   450,   254,    19,   253,     0,    51,    94,     0,   456,
       0,    92,     0,   216,   280,   272,     0,     0,     0,   305,
     313,     0,   316,     0,     0,   238,   246,   248,     0,   249,
       0,   247,     0,     0,   256,   261,   262,   263,   260,   430,
       0,    85,    85,   452,     0,   293,   417,   416,     0,   457,
     448,     0,   433,   432,   431,     0,    85,     0,    86,     0,
       0,     0,    77,    76,    81,     0,   140,     0,     0,   138,
       0,   150,     0,   171,     0,     0,   169,     0,     0,   191,
     192,    85,   226,   228,     0,     0,   224,     0,   209,     0,
     422,   420,     0,   396,     0,   438,     0,   359,   358,   357,
     352,   351,   355,   354,   353,   356,   361,   360,    31,     0,
       0,     0,     0,    96,     0,   267,     0,     0,     0,   304,
     277,   273,   274,   303,     0,   318,   317,   232,   231,     0,
       0,   240,     0,     0,   241,     0,     0,   429,     0,     0,
       0,    66,     0,     0,   418,     0,   218,     0,    56,     0,
      79,     0,     0,     0,    85,    85,   139,     0,   163,   161,
     159,   160,     0,   157,   154,     0,     0,   170,     0,   183,
     184,     0,   180,     0,     0,   195,   202,   203,     0,     0,
     200,     0,   193,     0,   206,     0,     0,     0,   208,   212,
       0,   397,   402,   437,   436,     0,    52,     0,     0,    95,
     102,     0,    93,   270,   269,   282,     0,     0,     0,     0,
       0,   283,   286,   281,   285,   284,   266,     0,   276,     0,
     307,     0,   308,   311,   310,   309,   306,   250,   251,     0,
       0,     0,     0,   428,   425,   435,     0,    67,   295,     0,
       0,   297,   294,     0,   458,   427,    80,    84,    83,    69,
       0,     0,   145,   147,     0,   141,   143,     0,   134,    85,
       0,   151,   176,   178,   172,   174,   182,   165,   199,     0,
     197,     0,     0,   187,   222,   214,     0,   423,    32,     0,
       0,     0,   108,     0,     0,   109,   110,   111,    98,   101,
       0,   100,     0,     0,   265,   287,     0,   278,     0,   275,
     302,   242,   244,   243,   245,    63,   299,     0,   300,   298,
     292,   419,    82,    85,     0,   162,    85,     0,   158,     0,
     156,    85,     0,    85,     0,   181,   196,   201,     0,   210,
       0,   121,     0,     0,   125,     0,     0,   129,     0,     0,
     107,   105,   104,   271,     0,   220,     0,   279,     0,     0,
     148,     0,   144,     0,   179,     0,   175,   198,   124,    85,
     123,   128,    85,   127,   132,    85,   131,    99,   290,    85,
       0,   296,     0,     0,     0,     0,   289,     0,     0,     0,
       0,   122,   126,   130,   288
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,     2,    63,    64,   293,    65,   508,    67,   124,
      68,    69,   224,    70,    71,    72,   368,   686,    73,   229,
      74,    75,   511,   604,   512,   513,   605,   514,   394,    76,
      77,    78,   564,   561,   649,   457,   741,   733,   734,    79,
      80,    81,   735,   809,   736,   812,   737,   815,    82,   231,
      83,   396,   519,   766,   767,   763,   764,   520,   616,   521,
     711,   612,   613,    84,   232,    85,   397,   526,   773,   774,
     771,   772,   621,   622,    86,   233,    87,   528,   529,   530,
     531,   629,   630,    88,    89,    90,   637,    91,   537,    92,
     384,   385,   235,   403,   404,   236,   237,    93,    94,    95,
     169,    96,    97,   173,    98,   176,   177,    99,   325,   464,
     465,   742,   468,   571,   572,   668,   567,   663,   664,   795,
     665,   100,   370,   592,   692,   759,   101,   327,   469,   574,
     676,   102,   157,   332,   333,   103,   104,   294,   105,   106,
     439,   107,   108,   109,   640,   110,   180,   111,   198,   359,
     386,   112,   181,   113,   114,   115,   116,   117,   186,   366,
     139,   197
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -396
static const yytype_int16 yypact[] =
{
    -396,   109,   784,  -396,   156,  -396,  -396,  -396,    -9,  -396,
    -396,   207,    23,  3670,   452,   438,  3741,   354,  3812,   215,
    3883,  -396,  -396,   264,  3954,   405,   466,  3457,  -396,   274,
    4025,   495,   350,   356,   501,   211,  -396,  4096,  5428,   298,
     253,  -396,  -396,  -396,  5783,  5285,  -396,  5783,  3528,  5783,
    5783,  5783,  3599,  5783,  5783,  5783,  5783,  5783,  5783,   300,
    5783,  5783,   502,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  3292,  -396,  -396,  -396,  3292,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
     182,  3292,    13,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  4828,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,   212,  -396,    15,
    5783,  -396,   332,  -396,    72,   276,     8,   360,  -396,  4654,
     440,  -396,   444,  -396,   451,   265,  4726,   479,   409,   193,
     492,  4879,   530,   543,  4930,   549,  -396,  3292,   550,  4981,
     556,  -396,   557,  -396,   558,  -396,  5032,   520,   559,   560,
     561,   564,  6324,   566,   572,   454,   574,  -396,  -396,   111,
     575,   121,    21,   128,   577,   508,   117,  -396,   582,  -396,
     202,   202,   583,  5083,  -396,  6324,  3363,   587,  -396,  3292,
    -396,  6016,  5499,  -396,   537,  5843,    84,    86,    80,  6520,
     592,  -396,  6324,   141,   442,   442,   151,   593,  -396,   149,
     151,   151,   151,   151,   151,   151,  -396,  -396,  -396,   434,
     151,   151,  -396,  -396,  -396,   596,   597,   214,  -396,  -396,
    -396,  -396,  -396,  -396,   358,  -396,  -396,  -396,  -396,   598,
     127,  -396,  5570,  5357,   594,  5783,  5783,  5783,  5783,  5783,
    5783,  5783,  5783,  5783,  5783,  5783,  5783,  4167,  5783,  5783,
    5783,  5783,  5783,  5783,  5783,  5783,   600,  5783,  5783,   599,
     599,   599,   599,   599,   599,   599,   599,   599,   599,   599,
    -396,  -396,  -396,  5783,  5783,   601,  -396,  -396,  -396,  -396,
    -396,  -396,   595,  -396,   607,  6324,  -396,  -396,   609,  5783,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  5783,   609,  4238,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,   222,  -396,   341,  -396,  -396,
    -396,  -396,   152,    62,  -396,  -396,  -396,  -396,  -396,  -396,
      54,  -396,  -396,   610,  -396,   614,    81,   107,   626,  -396,
     291,   625,  -396,    96,  -396,   628,  -396,   632,   631,   182,
     182,  -396,  -396,  -396,  -396,  -396,  5783,  -396,  -396,  -396,
     635,  -396,  -396,  6067,  -396,  5641,  5783,  -396,  -396,   585,
    -396,  5783,   578,  -396,   134,  -396,   182,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  1810,  1468,   548,   565,  1582,   302,
    -396,  -396,  1924,  -396,  3292,  -396,  -396,   116,  -396,   136,
    5783,  5904,  -396,  6324,  6324,  6324,  6324,  6324,  6324,  6324,
    6324,  6324,  6324,  6324,   391,  -396,   604,  6471,  6520,   221,
     221,   221,   221,   221,   221,  -396,   442,   442,  -396,  5783,
    5783,  5783,  5783,  5783,  5783,  5783,  5783,  5783,  5783,  5783,
    4777,  6373,  -396,  -396,  -396,   539,  6324,  -396,  6118,  -396,
     639,  6324,   640,   631,  -396,   612,   645,   643,   648,  -396,
    -396,   521,  -396,   649,   650,  -396,  -396,  -396,   652,  -396,
     654,  -396,    57,    61,  -396,  -396,  -396,  -396,  -396,  -396,
     137,  -396,  -396,  6324,  2038,  -396,  -396,  -396,  5965,  6324,
    -396,  6171,  -396,  -396,  -396,   631,  -396,   651,  -396,   522,
    4309,   655,  -396,  -396,  -396,   660,  -396,    70,   364,  -396,
     656,  -396,   663,  -396,    87,   670,  -396,   112,   671,   666,
    -396,  -396,  -396,  -396,   678,  2152,  -396,   642,  -396,   328,
    -396,  -396,  6222,  -396,  5783,  -396,  4380,  1434,  1434,  1548,
     746,   746,   227,   227,   227,   235,   151,   151,  -396,  5783,
    5783,   367,  4451,  -396,   367,  -396,   142,   406,   682,  -396,
     658,   633,  -396,  -396,   567,  -396,  -396,  -396,  -396,   706,
     708,  -396,   709,   712,  -396,   713,   714,  -396,   711,  2266,
    2380,  5783,   340,  5783,  -396,  5783,  -396,  2494,  -396,   718,
    -396,   719,  5134,   720,  -396,  -396,  -396,   401,  -396,  -396,
    -396,   653,   218,  -396,  -396,   721,   414,  -396,   420,  -396,
    -396,   219,  -396,   722,   723,  -396,  -396,  -396,   609,    44,
    -396,   724,  -396,  1696,  -396,   727,   690,   680,  -396,  -396,
     681,  -396,   659,  -396,  6422,   196,  6324,   898,  3292,  -396,
    -396,  4582,  -396,  -396,  -396,  -396,   158,   734,   735,   733,
     736,  -396,  -396,  -396,  -396,  -396,  -396,  5712,  -396,   643,
    -396,   737,  -396,  -396,  -396,  -396,  -396,  -396,  -396,   738,
     740,   741,   742,  -396,  -396,  -396,   743,  6324,  -396,   217,
     744,  -396,  -396,  6273,  6324,  -396,  -396,  -396,  -396,  -396,
    2608,  1468,  -396,  -396,     5,  -396,  -396,    90,  -396,  -396,
    3292,  -396,  -396,  -396,  -396,  -396,   525,  -396,  -396,   745,
    -396,   552,   609,  -396,  -396,  -396,   747,  -396,  -396,   365,
     402,   415,  -396,   749,   898,  -396,  -396,  -396,  -396,  -396,
    4522,  -396,   695,  5783,  -396,  -396,   677,  -396,    18,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,    75,  -396,  -396,
    -396,  -396,  -396,  -396,  3292,  -396,  -396,  3292,  -396,  2722,
    -396,  -396,  3292,  -396,  3292,  -396,  -396,  -396,   750,  -396,
     751,  -396,  3292,   753,  -396,  3292,   756,  -396,  3292,   757,
    -396,  -396,  6324,  -396,  5185,   182,    75,  -396,   197,  1012,
    -396,  1126,  -396,  1240,  -396,  1354,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
     758,  -396,  2836,  2950,  3064,  3178,  -396,   760,   761,   763,
     764,  -396,  -396,  -396,  -396
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -396,  -396,  -396,  -396,  -396,  -332,  -396,    -2,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,    51,  -396,  -396,  -396,  -396,  -396,  -179,  -396,
    -396,  -396,  -396,  -396,   204,  -396,  -396,    35,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,   374,  -396,  -396,
    -396,  -396,    66,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,    58,  -396,  -396,  -396,  -396,  -396,   246,
    -396,  -396,    55,  -396,  -242,  -396,  -396,  -396,  -396,  -396,
    -235,   272,  -329,  -396,  -396,  -396,  -396,  -396,  -396,  -396,
    -396,  -396,  -395,  -396,  -396,  -396,   423,  -396,  -396,  -396,
    -396,  -396,   314,  -396,   113,  -396,  -396,  -396,   224,  -396,
     226,  -396,  -396,  -396,  -396,   -17,  -396,  -396,  -396,  -396,
    -396,  -396,  -396,  -396,   334,  -396,  -396,  -337,   -10,  -396,
     349,   -12,   -37,   778,  -396,  -396,  -396,  -396,  -396,   646,
    -396,  -396,  -396,  -396,  -396,  -396,  -396,   -35,  -396,  -396,
    -396,  -396
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -454
static const yytype_int16 yytable[] =
{
      66,   129,   125,   474,   136,   407,   141,   138,   144,   286,
     196,   300,   149,   203,   239,   156,   285,   209,   162,   286,
     287,   487,   288,   289,   122,   183,   185,   346,   347,   123,
     491,   492,   191,   195,   119,   199,   202,   204,   205,   206,
     202,   210,   211,   212,   213,   214,   215,   720,   220,   221,
     395,   301,   223,   120,   398,   473,   402,   506,   286,   287,
     581,   288,   289,   472,   584,  -314,   290,   291,   240,   228,
     282,   607,   797,   230,   286,   297,   608,   609,   610,   286,
     287,   382,   288,   289,   477,   377,   383,   379,   618,   238,
    -182,   619,   282,   620,   286,   284,   608,   609,   610,   282,
     286,   292,   485,   486,   282,   290,   291,   282,   295,     3,
     479,   292,   282,   624,   342,   625,   626,   538,   627,   282,
     354,   721,   345,   490,  -255,   282,   290,   291,   406,   348,
    -182,   349,   582,   383,   722,   503,   585,   540,   587,  -314,
     378,  -216,   380,   653,   388,   321,   282,   583,   282,   298,
     292,   586,   390,  -255,   282,   470,   478,  -216,   282,   118,
     350,   284,   282,   381,  -182,   282,   292,   282,   282,   282,
     539,   292,   662,   282,   282,   282,   282,   282,   282,   673,
     373,  -216,   480,   282,   282,   611,   292,   369,   343,   494,
     541,   588,   292,   505,   355,   504,   654,   691,  -255,   728,
     756,   119,   628,   357,  -216,   351,   242,   409,   243,   244,
     121,   505,   178,   284,   505,   178,   142,   179,   284,   505,
     756,   705,   714,   462,   535,  -268,   284,   234,   566,   471,
     202,   411,   743,   413,   414,   415,   416,   417,   418,   419,
     420,   421,   422,   423,   424,   426,   427,   428,   429,   430,
     431,   432,   433,   434,  -268,   436,   437,   358,   282,   190,
     119,   706,   715,   280,   281,   145,  -424,   146,   307,  -424,
     313,   450,   451,   284,   758,   158,   242,   463,   243,   244,
     159,   160,   242,   314,   243,   244,   283,   456,   455,   284,
     242,   757,   243,   244,   758,   707,   716,   482,   483,   187,
     458,   188,   461,   459,   216,   532,   217,   147,   308,   266,
     267,   268,   589,   590,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   661,   218,   597,   277,   278,
     279,   638,   672,   280,   281,   296,   282,   278,   279,   280,
     281,   189,   466,   688,  -272,   533,   689,   280,   281,   690,
     299,   166,   633,   167,   493,   137,   168,   170,   219,   399,
     123,   400,   171,   498,   499,   614,   780,  -153,   781,   501,
     647,   639,   765,   467,   282,   611,   282,   282,   282,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   172,   282,
     282,   282,   282,   282,   282,   282,   282,   282,   542,   282,
     282,   401,   536,   783,   702,   784,   150,  -153,   782,   655,
     648,   151,   656,   282,   282,   657,   786,   709,   787,   282,
     798,   282,   120,   712,   282,   700,   701,   547,   548,   549,
     550,   551,   552,   553,   554,   555,   556,   557,   391,   132,
     392,   133,   658,   304,   703,   785,   242,   305,   243,   244,
     659,   660,   134,   130,   306,   131,   282,   710,   788,   820,
     393,   282,   282,   713,   282,   256,   819,   152,   544,   257,
     258,   259,   153,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   311,   312,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   315,   164,   242,   602,   243,
     244,   165,   174,   280,   281,   282,   222,   175,   123,   642,
     282,   282,   282,   282,   282,   282,   282,   282,   282,   282,
     282,   330,   575,   599,   645,   600,   331,   331,   340,   619,
     769,   620,   202,   317,   644,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   318,   202,   646,   515,
     651,   516,   320,   322,   280,   281,   626,  -149,   627,   324,
     326,   328,   334,   335,   336,   282,   522,   337,   523,   338,
     670,   517,   518,   656,  -149,   339,   671,   341,   344,   687,
     352,   693,   353,   694,   799,   356,   361,   801,   524,   518,
     367,  -152,   803,   374,   805,   387,   389,   150,   152,   453,
     412,   405,   438,   658,   452,   545,   435,   282,  -152,   282,
     454,   659,   660,   560,   282,   123,   475,   476,   719,   440,
     441,   442,   443,   444,   445,   446,   447,   448,   449,   481,
     822,   484,   748,   823,   175,   489,   824,   383,   495,   502,
     825,   500,   563,   565,   467,   732,   738,   546,   569,   570,
     282,   573,   577,   578,   598,   202,   282,   282,   579,   242,
     580,   243,   244,   606,   603,   615,   617,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   623,
     631,   634,   257,   258,   259,   666,   260,   261,   262,   263,
     264,   265,   266,   267,   268,   527,   636,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   770,   677,
     669,   678,   778,   667,   683,   679,   280,   281,   680,   681,
     682,   696,   697,   699,   708,   717,   718,   723,   792,   704,
     724,   794,   732,   725,   726,   727,   284,   744,   745,   179,
     750,   751,   746,   752,   753,   754,   755,   760,   776,   793,
     779,   796,   762,   807,   808,   282,   811,   282,   789,   814,
     817,   826,   800,   831,   832,   802,   833,   834,   652,   790,
     804,   525,   806,   768,   775,   632,   777,   596,   488,   568,
     810,   821,   749,   813,    -2,     4,   816,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,   674,    16,
     675,   242,    17,   243,   244,   576,    18,    19,   163,    20,
      21,    22,    23,     0,    24,    25,     0,    26,    27,    28,
       0,    29,    30,    31,    32,    33,    34,   360,    35,     0,
      36,    37,    38,    39,    40,    41,    42,    43,     0,    44,
       0,    45,     0,    46,   274,   275,   276,   277,   278,   279,
       0,     0,     0,     0,     0,     0,     0,     0,   280,   281,
       0,     0,     0,     0,     0,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,    52,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     4,
       0,     5,     6,     7,     8,     9,    10,  -106,    12,    13,
      14,    15,     0,    16,     0,     0,    17,   729,   730,   731,
      18,     0,     0,    20,    21,    22,    23,     0,    24,   225,
       0,   226,    27,    28,     0,     0,    30,     0,     0,     0,
       0,     0,   227,     0,    36,    37,    38,    39,     0,    41,
      42,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -146,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,  -146,  -146,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,  -146,   227,     0,    36,    37,
      38,    39,     0,    41,    42,    43,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,    52,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     4,     0,     5,
       6,     7,     8,     9,    10,  -142,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,  -142,
    -142,    20,    21,    22,    23,     0,    24,   225,     0,   226,
      27,    28,     0,     0,    30,     0,     0,     0,     0,  -142,
     227,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,  -177,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,  -177,  -177,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,  -177,   227,     0,    36,    37,    38,    39,
       0,    41,    42,    43,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,    52,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     4,     0,     5,     6,     7,
       8,     9,    10,  -173,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,  -173,  -173,    20,
      21,    22,    23,     0,    24,   225,     0,   226,    27,    28,
       0,     0,    30,     0,     0,     0,     0,  -173,   227,     0,
      36,    37,    38,    39,     0,    41,    42,    43,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,    52,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     4,
       0,     5,     6,     7,     8,     9,    10,   -75,    12,    13,
      14,    15,     0,    16,   509,   510,    17,     0,     0,   242,
      18,   243,   244,    20,    21,    22,    23,     0,    24,   225,
       0,   226,    27,    28,     0,     0,    30,     0,     0,     0,
       0,     0,   227,     0,    36,    37,    38,    39,     0,    41,
      42,    43,     0,    44,     0,    45,     0,    46,     0,   271,
     272,   273,   274,   275,   276,   277,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,   280,   281,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -190,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,   242,    18,   243,   244,    20,    21,    22,
      23,   527,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,    43,     0,    44,     0,    45,
       0,    46,     0,     0,   272,   273,   274,   275,   276,   277,
     278,   279,     0,     0,     0,     0,     0,     0,     0,     0,
     280,   281,     0,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,    52,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     4,     0,     5,
       6,     7,     8,     9,    10,  -194,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,  -194,    24,   225,     0,   226,
      27,    28,     0,     0,    30,     0,     0,     0,     0,     0,
     227,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   507,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,    43,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,    52,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     4,     0,     5,     6,     7,
       8,     9,    10,   534,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,     0,    24,   225,     0,   226,    27,    28,
       0,     0,    30,     0,     0,     0,     0,     0,   227,     0,
      36,    37,    38,    39,     0,    41,    42,    43,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,    52,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     4,
       0,     5,     6,     7,     8,     9,    10,   591,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,     0,    24,   225,
       0,   226,    27,    28,     0,     0,    30,     0,     0,     0,
       0,     0,   227,     0,    36,    37,    38,    39,     0,    41,
      42,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,   635,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,    43,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,    52,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     4,     0,     5,
       6,     7,     8,     9,    10,   684,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,     0,    24,   225,     0,   226,
      27,    28,     0,     0,    30,     0,     0,     0,     0,     0,
     227,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   685,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,    43,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,    52,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     4,     0,     5,     6,     7,
       8,     9,    10,     0,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,     0,    24,   225,     0,   226,    27,    28,
       0,     0,    30,     0,     0,     0,     0,     0,   227,     0,
      36,    37,    38,    39,     0,    41,    42,    43,     0,    44,
       0,    45,     0,    46,   695,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,    52,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     4,
       0,     5,     6,     7,     8,     9,    10,   -78,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,     0,    24,   225,
       0,   226,    27,    28,     0,     0,    30,     0,     0,     0,
       0,     0,   227,     0,    36,    37,    38,    39,     0,    41,
      42,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,  -155,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,    43,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,    52,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,     4,     0,     5,
       6,     7,     8,     9,    10,   827,    12,    13,    14,    15,
       0,    16,     0,     0,    17,     0,     0,     0,    18,     0,
       0,    20,    21,    22,    23,     0,    24,   225,     0,   226,
      27,    28,     0,     0,    30,     0,     0,     0,     0,     0,
     227,     0,    36,    37,    38,    39,     0,    41,    42,    43,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,     0,     0,
       0,    48,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,     0,     0,     0,     0,     0,    52,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,     4,     0,     5,     6,     7,     8,     9,    10,   828,
      12,    13,    14,    15,     0,    16,     0,     0,    17,     0,
       0,     0,    18,     0,     0,    20,    21,    22,    23,     0,
      24,   225,     0,   226,    27,    28,     0,     0,    30,     0,
       0,     0,     0,     0,   227,     0,    36,    37,    38,    39,
       0,    41,    42,    43,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,    48,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,     0,     0,     0,
       0,     0,    52,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,     4,     0,     5,     6,     7,
       8,     9,    10,   829,    12,    13,    14,    15,     0,    16,
       0,     0,    17,     0,     0,     0,    18,     0,     0,    20,
      21,    22,    23,     0,    24,   225,     0,   226,    27,    28,
       0,     0,    30,     0,     0,     0,     0,     0,   227,     0,
      36,    37,    38,    39,     0,    41,    42,    43,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,     0,     0,     0,    48,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,     0,     0,     0,     0,     0,    52,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,     4,
       0,     5,     6,     7,     8,     9,    10,   830,    12,    13,
      14,    15,     0,    16,     0,     0,    17,     0,     0,     0,
      18,     0,     0,    20,    21,    22,    23,     0,    24,   225,
       0,   226,    27,    28,     0,     0,    30,     0,     0,     0,
       0,     0,   227,     0,    36,    37,    38,    39,     0,    41,
      42,    43,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
       0,     0,     0,    48,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,     0,     0,     0,     0,     0,
      52,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,     4,     0,     5,     6,     7,     8,     9,
      10,     0,    12,    13,    14,    15,     0,    16,     0,     0,
      17,     0,     0,     0,    18,     0,     0,    20,    21,    22,
      23,     0,    24,   225,     0,   226,    27,    28,     0,     0,
      30,     0,     0,     0,     0,     0,   227,     0,    36,    37,
      38,    39,     0,    41,    42,    43,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   363,     0,     0,  -453,  -453,  -453,
    -453,  -453,     0,    47,     0,     0,     0,    48,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,  -453,
    -453,     0,     0,     0,    52,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,  -453,     0,  -453,
       0,  -453,     0,     0,  -453,  -453,     0,     0,  -453,   364,
    -453,     0,  -453,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     365,     0,     0,     0,  -453,     0,     0,     0,     0,     0,
       0,     0,     0,     0,  -453,  -453,     0,     0,   154,  -453,
     155,     6,     7,   127,     9,    10,     0,  -453,  -453,  -453,
    -453,  -453,  -453,     0,  -453,  -453,  -453,  -453,     0,     0,
       0,     0,     0,    21,    22,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   200,
       0,   201,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     207,     0,   208,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,    44,     0,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   126,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   135,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   140,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   143,     0,     0,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   148,     0,     0,     6,     7,
     127,     9,    10,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
      21,    22,     0,     0,     0,     0,     0,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   128,     0,
      36,     0,    38,     0,     0,    41,    42,     0,     0,    44,
       0,    45,     0,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   161,     0,     0,     6,
       7,   127,     9,    10,     0,    47,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,    50,     0,     0,     0,
      51,    21,    22,     0,     0,     0,     0,     0,    53,    54,
      55,    56,    57,    58,     0,    59,    60,    61,    62,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
      44,     0,    45,     0,    46,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   182,     0,     0,
       6,     7,   127,     9,    10,     0,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,    21,    22,     0,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     128,     0,    36,     0,    38,     0,     0,    41,    42,     0,
       0,    44,     0,    45,     0,    46,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   425,     0,
       0,     6,     7,   127,     9,    10,     0,    47,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,    50,     0,
       0,     0,    51,    21,    22,     0,     0,     0,     0,     0,
      53,    54,    55,    56,    57,    58,     0,    59,    60,    61,
      62,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   460,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,     0,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,     0,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     601,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,    44,     0,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   643,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,     0,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   650,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,    44,     0,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   791,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,     0,    44,     0,    45,
       0,    46,     0,   739,     0,  -103,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,     0,
       0,     0,     0,     0,     0,  -103,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   242,     0,   243,
     244,     0,     0,     0,     0,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   302,     0,   740,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     0,     0,     0,     0,
       0,     0,     0,     0,   280,   281,     0,   303,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   242,
       0,   243,   244,     0,     0,     0,     0,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   309,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,     0,     0,
       0,     0,     0,     0,     0,     0,   280,   281,     0,   310,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     558,   242,     0,   243,   244,     0,     0,     0,     0,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
       0,   241,   242,     0,   243,   244,     0,     0,   280,   281,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,   559,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,     0,   316,   242,     0,   243,   244,     0,     0,   280,
     281,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,     0,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,     0,   319,   242,     0,   243,   244,     0,     0,
     280,   281,   245,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,     0,     0,     0,   257,   258,   259,
       0,   260,   261,   262,   263,   264,   265,   266,   267,   268,
       0,     0,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   279,     0,   323,   242,     0,   243,   244,     0,
       0,   280,   281,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,     0,     0,     0,   257,   258,
     259,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,     0,   329,   242,     0,   243,   244,
       0,     0,   280,   281,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,     0,     0,     0,   257,
     258,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,     0,   362,   242,     0,   243,
     244,     0,     0,   280,   281,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,     0,     0,     0,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     0,   698,   242,     0,
     243,   244,     0,     0,   280,   281,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,     0,   818,   242,
       0,   243,   244,     0,     0,   280,   281,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,     0,     0,
     242,     0,   243,   244,     0,     0,   280,   281,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     6,
       7,   127,     9,    10,     0,     0,     0,   280,   281,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,    22,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   192,   128,
       0,    36,     0,    38,     0,     0,    41,    42,     0,     0,
      44,   193,    45,     0,    46,     0,   194,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     7,   127,     9,    10,    47,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,    50,     0,     0,
       0,    51,     0,    21,    22,     0,     0,     0,     0,    53,
      54,    55,    56,    57,    58,     0,    59,    60,    61,    62,
     192,   128,     0,    36,     0,    38,     0,     0,    41,    42,
       0,     0,    44,     0,    45,     0,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     6,     7,   127,     9,    10,     0,    47,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,    50,
       0,     0,     0,    51,    21,    22,     0,   410,     0,     0,
       0,    53,    54,    55,    56,    57,    58,     0,    59,    60,
      61,    62,   128,     0,    36,     0,    38,     0,     0,    41,
      42,     0,     0,    44,   184,    45,     0,    46,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     6,     7,   127,     9,    10,     0,    47,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
      50,     0,     0,     0,    51,    21,    22,     0,     0,     0,
       0,     0,    53,    54,    55,    56,    57,    58,     0,    59,
      60,    61,    62,   128,     0,    36,     0,    38,     0,     0,
      41,    42,     0,     0,    44,   372,    45,     0,    46,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     6,     7,   127,     9,    10,     0,
      47,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,    50,     0,     0,     0,    51,    21,    22,     0,     0,
       0,     0,     0,    53,    54,    55,    56,    57,    58,     0,
      59,    60,    61,    62,   128,     0,    36,     0,    38,     0,
       0,    41,    42,     0,   408,    44,     0,    45,     0,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     7,   127,     9,    10,
       0,    47,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,    50,     0,     0,     0,    51,    21,    22,     0,
       0,     0,     0,     0,    53,    54,    55,    56,    57,    58,
       0,    59,    60,    61,    62,   128,     0,    36,     0,    38,
       0,     0,    41,    42,     0,     0,    44,   497,    45,     0,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     6,     7,   127,     9,
      10,     0,    47,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,    50,     0,     0,     0,    51,    21,    22,
       0,     0,     0,     0,     0,    53,    54,    55,    56,    57,
      58,     0,    59,    60,    61,    62,   128,     0,    36,     0,
      38,     0,     0,    41,    42,     0,   747,    44,     0,    45,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     6,     7,   127,
       9,    10,     0,    47,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,    50,     0,     0,     0,    51,    21,
      22,     0,     0,     0,     0,     0,    53,    54,    55,    56,
      57,    58,     0,    59,    60,    61,    62,   128,     0,    36,
       0,    38,     0,     0,    41,    42,     0,     0,    44,     0,
      45,     0,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    47,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,    50,     0,     0,     0,    51,
       0,     0,     0,     0,     0,     0,   375,    53,    54,    55,
      56,    57,    58,     0,    59,    60,    61,    62,   242,     0,
     243,   244,     0,     0,   376,     0,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   375,     0,     0,
       0,     0,     0,     0,     0,   280,   281,     0,     0,   242,
     543,   243,   244,     0,     0,     0,     0,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   593,     0,
       0,     0,     0,     0,     0,     0,   280,   281,     0,     0,
     242,   594,   243,   244,     0,     0,     0,     0,   245,   246,
     247,   248,   249,   250,   251,   252,   253,   254,   255,   256,
       0,     0,     0,   257,   258,   259,     0,   260,   261,   262,
     263,   264,   265,   266,   267,   268,     0,     0,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     0,
     371,   242,     0,   243,   244,     0,     0,   280,   281,   245,
     246,   247,   248,   249,   250,   251,   252,   253,   254,   255,
     256,     0,     0,     0,   257,   258,   259,     0,   260,   261,
     262,   263,   264,   265,   266,   267,   268,     0,     0,   269,
     270,   271,   272,   273,   274,   275,   276,   277,   278,   279,
       0,     0,   242,   496,   243,   244,     0,     0,   280,   281,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,     0,     0,     0,   257,   258,   259,     0,   260,
     261,   262,   263,   264,   265,   266,   267,   268,     0,     0,
     269,   270,   271,   272,   273,   274,   275,   276,   277,   278,
     279,     0,     0,   242,     0,   243,   244,     0,     0,   280,
     281,   245,   246,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,     0,   562,     0,   257,   258,   259,     0,
     260,   261,   262,   263,   264,   265,   266,   267,   268,     0,
       0,   269,   270,   271,   272,   273,   274,   275,   276,   277,
     278,   279,     0,     0,     0,     0,   242,     0,   243,   244,
     280,   281,   595,     0,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,     0,     0,     0,   257,
     258,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,     0,     0,   242,   641,   243,
     244,     0,     0,   280,   281,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,     0,     0,     0,
     257,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,     0,     0,   242,   761,
     243,   244,     0,     0,   280,   281,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,     0,     0,   242,
       0,   243,   244,     0,     0,   280,   281,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,     0,
       0,     0,   257,   258,   259,     0,   260,   261,   262,   263,
     264,   265,   266,   267,   268,     0,     0,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,   242,     0,
     243,   244,     0,     0,     0,     0,   280,   281,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   257,   258,   259,     0,   260,   261,   262,   263,   264,
     265,   266,   267,   268,     0,     0,   269,   270,   271,   272,
     273,   274,   275,   276,   277,   278,   279,   242,     0,   243,
     244,     0,     0,     0,     0,   280,   281,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   258,   259,     0,   260,   261,   262,   263,   264,   265,
     266,   267,   268,     0,     0,   269,   270,   271,   272,   273,
     274,   275,   276,   277,   278,   279,   242,     0,   243,   244,
       0,     0,     0,     0,   280,   281,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   259,     0,   260,   261,   262,   263,   264,   265,   266,
     267,   268,     0,     0,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   242,     0,   243,   244,     0,
       0,     0,     0,   280,   281,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   260,   261,   262,   263,   264,   265,   266,   267,
     268,     0,     0,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,     0,     0,     0,     0,     0,     0,
       0,     0,   280,   281
};

static const yytype_int16 yycheck[] =
{
       2,    13,    12,   340,    16,   240,    18,    17,    20,     4,
      45,     3,    24,    48,     1,    27,     1,    52,    30,     4,
       5,   353,     7,     8,     1,    37,    38,     6,     7,     6,
     359,   360,    44,    45,    43,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,     3,    60,    61,
     229,    43,    62,    62,   233,     1,   235,   386,     4,     5,
       3,     7,     8,     1,     3,     3,    51,    52,    55,    71,
     107,     1,    54,    75,     4,     3,     6,     7,     8,     4,
       5,     1,     7,     8,     3,     1,     6,     1,     1,    91,
       3,     4,   129,     6,     4,    77,     6,     7,     8,   136,
       4,    96,     6,     7,   141,    51,    52,   144,   120,     0,
       3,    96,   149,     1,     3,     3,     4,     1,     6,   156,
       3,    77,     1,   358,     3,   162,    51,    52,     1,     1,
      43,     3,    75,     6,    90,     1,    75,     1,     1,    77,
      56,    61,    56,     1,     3,   147,   183,    90,   185,    77,
      96,    90,     3,    32,   191,     3,    75,    77,   195,     3,
      32,    77,   199,    77,    77,   202,    96,   204,   205,   206,
      54,    96,   567,   210,   211,   212,   213,   214,   215,   574,
     192,    54,    75,   220,   221,   517,    96,   189,    77,   368,
      54,    54,    96,    77,    77,    61,    54,   592,    77,     3,
       3,    43,    90,     1,    77,    77,    55,   242,    57,    58,
       3,    77,     1,    77,    77,     1,     1,     6,    77,    77,
       3,     3,     3,     1,   403,     3,    77,    45,   463,    77,
     242,   243,    74,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   263,   264,   265,    32,   267,   268,    55,   295,     6,
      43,    43,    43,   112,   113,     1,    55,     3,     3,    55,
      77,   283,   284,    77,    77,     1,    55,    55,    57,    58,
       6,     7,    55,    90,    57,    58,    74,   299,   298,    77,
      55,    74,    57,    58,    77,    77,    77,     6,     7,     1,
     312,     3,   314,   313,     4,     3,     6,    43,    43,    88,
      89,    90,   491,   492,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   567,    26,   506,   101,   102,
     103,     3,   574,   112,   113,     3,   373,   102,   103,   112,
     113,    43,     1,     3,     3,    43,     6,   112,   113,     9,
      74,     1,   531,     3,   366,     1,     6,     1,    58,     1,
       6,     3,     6,   375,   376,     1,     1,     3,     3,   381,
       3,    43,   704,    32,   411,   707,   413,   414,   415,   416,
     417,   418,   419,   420,   421,   422,   423,   424,    32,   426,
     427,   428,   429,   430,   431,   432,   433,   434,   410,   436,
     437,    43,   404,     1,     3,     3,     1,    43,    43,     3,
      43,     6,     6,   450,   451,     9,     1,     3,     3,   456,
     757,   458,    62,     3,   461,   604,   605,   439,   440,   441,
     442,   443,   444,   445,   446,   447,   448,   449,     4,     1,
       6,     3,    36,     3,    43,    43,    55,     3,    57,    58,
      44,    45,    14,     1,     3,     3,   493,    43,    43,   796,
      26,   498,   499,    43,   501,    74,   795,     1,    77,    78,
      79,    80,     6,    82,    83,    84,    85,    86,    87,    88,
      89,    90,     3,    74,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,     3,     1,    55,   510,    57,
      58,     6,     1,   112,   113,   542,     4,     6,     6,   544,
     547,   548,   549,   550,   551,   552,   553,   554,   555,   556,
     557,     1,     1,     1,   559,     3,     6,     6,    74,     4,
     709,     6,   544,     3,   546,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,     3,   559,   560,     1,
     562,     3,     3,     3,   112,   113,     4,     9,     6,     3,
       3,     3,     3,     3,     3,   602,     1,     3,     3,     3,
       3,    23,    24,     6,     9,     3,     9,     3,     3,   591,
       3,   593,    74,   595,   763,     3,     3,   766,    23,    24,
       3,    43,   771,    56,   773,     3,     3,     1,     1,     4,
       6,     3,     3,    36,     3,     1,     6,   644,    43,   646,
       3,    44,    45,    74,   651,     6,     6,     3,   628,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,     3,
     809,     6,   667,   812,     6,     3,   815,     6,     3,    61,
     819,    56,     3,     3,    32,   647,   648,    43,     3,     6,
     687,     3,     3,     3,     3,   667,   693,   694,     6,    55,
       6,    57,    58,     3,     9,     9,     3,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,     9,
       9,     3,    78,    79,    80,     3,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    29,    54,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,   710,     3,
      77,     3,   722,    55,     3,     6,   112,   113,     6,     6,
       6,     3,     3,     3,     3,     3,     3,     3,   740,    76,
       3,   743,   734,    43,    54,    54,    77,     3,     3,     6,
       3,     3,     6,     3,     3,     3,     3,     3,     3,    54,
       3,    74,   701,     3,     3,   792,     3,   794,     9,     3,
       3,     3,   764,     3,     3,   767,     3,     3,   564,   734,
     772,   397,   774,   707,   716,   529,   721,   505,   355,   465,
     782,   798,   669,   785,     0,     1,   788,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,   574,    15,
     574,    55,    18,    57,    58,   471,    22,    23,    30,    25,
      26,    27,    28,    -1,    30,    31,    -1,    33,    34,    35,
      -1,    37,    38,    39,    40,    41,    42,   181,    44,    -1,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    98,    99,   100,   101,   102,   103,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   112,   113,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    19,    20,    21,
      22,    -1,    -1,    25,    26,    27,    28,    -1,    30,    31,
      -1,    33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,
      -1,    -1,    44,    -1,    46,    47,    48,    49,    -1,    51,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    23,    24,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    43,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    23,
      24,    25,    26,    27,    28,    -1,    30,    31,    -1,    33,
      34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    43,
      44,    -1,    46,    47,    48,    49,    -1,    51,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,
      -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,   107,   108,   109,    -1,   111,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    23,    24,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    43,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    23,    24,    25,
      26,    27,    28,    -1,    30,    31,    -1,    33,    34,    35,
      -1,    -1,    38,    -1,    -1,    -1,    -1,    43,    44,    -1,
      46,    47,    48,    49,    -1,    51,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    16,    17,    18,    -1,    -1,    55,
      22,    57,    58,    25,    26,    27,    28,    -1,    30,    31,
      -1,    33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,
      -1,    -1,    44,    -1,    46,    47,    48,    49,    -1,    51,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   112,   113,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    55,    22,    57,    58,    25,    26,    27,
      28,    29,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     112,   113,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    29,    30,    31,    -1,    33,
      34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,
      44,    -1,    46,    47,    48,    49,    -1,    51,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,
      -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,   107,   108,   109,    -1,   111,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    -1,    30,    31,    -1,    33,    34,    35,
      -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,    -1,
      46,    47,    48,    49,    -1,    51,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    -1,    30,    31,
      -1,    33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,
      -1,    -1,    44,    -1,    46,    47,    48,    49,    -1,    51,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    -1,    30,    31,    -1,    33,
      34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,
      44,    -1,    46,    47,    48,    49,    -1,    51,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,
      -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,   107,   108,   109,    -1,   111,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,    -1,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    -1,    30,    31,    -1,    33,    34,    35,
      -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,    -1,
      46,    47,    48,    49,    -1,    51,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    60,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    -1,    30,    31,
      -1,    33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,
      -1,    -1,    44,    -1,    46,    47,    48,    49,    -1,    51,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    -1,   102,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,     1,    -1,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    15,    -1,    -1,    18,    -1,    -1,    -1,    22,    -1,
      -1,    25,    26,    27,    28,    -1,    30,    31,    -1,    33,
      34,    35,    -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,
      44,    -1,    46,    47,    48,    49,    -1,    51,    52,    53,
      -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,
      -1,    85,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,
      -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,   102,    -1,
     104,   105,   106,   107,   108,   109,    -1,   111,   112,   113,
     114,     1,    -1,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    -1,    15,    -1,    -1,    18,    -1,
      -1,    -1,    22,    -1,    -1,    25,    26,    27,    28,    -1,
      30,    31,    -1,    33,    34,    35,    -1,    -1,    38,    -1,
      -1,    -1,    -1,    -1,    44,    -1,    46,    47,    48,    49,
      -1,    51,    52,    53,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    81,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    -1,    -1,    -1,
      -1,    -1,   102,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,     1,    -1,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    -1,    15,
      -1,    -1,    18,    -1,    -1,    -1,    22,    -1,    -1,    25,
      26,    27,    28,    -1,    30,    31,    -1,    33,    34,    35,
      -1,    -1,    38,    -1,    -1,    -1,    -1,    -1,    44,    -1,
      46,    47,    48,    49,    -1,    51,    52,    53,    -1,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    85,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    -1,    -1,    -1,    -1,    -1,   102,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,     1,
      -1,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    -1,    15,    -1,    -1,    18,    -1,    -1,    -1,
      22,    -1,    -1,    25,    26,    27,    28,    -1,    30,    31,
      -1,    33,    34,    35,    -1,    -1,    38,    -1,    -1,    -1,
      -1,    -1,    44,    -1,    46,    47,    48,    49,    -1,    51,
      52,    53,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,
      -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    -1,    -1,    -1,    -1,    -1,
     102,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,     1,    -1,     3,     4,     5,     6,     7,
       8,    -1,    10,    11,    12,    13,    -1,    15,    -1,    -1,
      18,    -1,    -1,    -1,    22,    -1,    -1,    25,    26,    27,
      28,    -1,    30,    31,    -1,    33,    34,    35,    -1,    -1,
      38,    -1,    -1,    -1,    -1,    -1,    44,    -1,    46,    47,
      48,    49,    -1,    51,    52,    53,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    81,    -1,    -1,    -1,    85,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,
      27,    -1,    -1,    -1,   102,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    55,    56,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,     1,    96,
       3,     4,     5,     6,     7,     8,    -1,   104,   105,   106,
     107,   108,   109,    -1,   111,   112,   113,   114,    -1,    -1,
      -1,    -1,    -1,    26,    27,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,     3,     4,     5,     6,     7,     8,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    26,    27,    -1,    -1,    -1,    -1,
      -1,   104,   105,   106,   107,   108,   109,    -1,   111,   112,
     113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     3,     4,     5,     6,     7,     8,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,    -1,
      -1,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,
      -1,    -1,    -1,   104,   105,   106,   107,   108,   109,    -1,
     111,   112,   113,   114,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,
      -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,
      -1,    -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,
     109,    -1,   111,   112,   113,   114,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,
      27,    -1,    -1,    -1,    -1,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,     5,
       6,     7,     8,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      26,    27,    -1,    -1,    -1,    -1,    -1,   104,   105,   106,
     107,   108,   109,    -1,   111,   112,   113,   114,    44,    -1,
      46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,    55,
      -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,     4,
       5,     6,     7,     8,    -1,    81,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,
      96,    26,    27,    -1,    -1,    -1,    -1,    -1,   104,   105,
     106,   107,   108,   109,    -1,   111,   112,   113,   114,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
       4,     5,     6,     7,     8,    -1,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    26,    27,    -1,    -1,    -1,    -1,    -1,   104,
     105,   106,   107,   108,   109,    -1,   111,   112,   113,   114,
      44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,
      -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
      -1,     4,     5,     6,     7,     8,    -1,    81,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,
      -1,    -1,    96,    26,    27,    -1,    -1,    -1,    -1,    -1,
     104,   105,   106,   107,   108,   109,    -1,   111,   112,   113,
     114,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    26,    27,    -1,    -1,    -1,    -1,
      -1,   104,   105,   106,   107,   108,   109,    -1,   111,   112,
     113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,    -1,
      -1,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    55,    -1,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,
      -1,    -1,    -1,   104,   105,   106,   107,   108,   109,    -1,
     111,   112,   113,   114,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    -1,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,
      -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    55,    -1,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,
      -1,    -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,
     109,    -1,   111,   112,   113,   114,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    -1,    55,    -1,    57,
      -1,    59,    -1,     1,    -1,     3,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    -1,
      -1,    -1,    -1,    -1,    -1,    43,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,    55,    -1,    57,
      58,    -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,     3,    -1,    77,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   112,   113,    -1,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    55,
      -1,    57,    58,    -1,    -1,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,     3,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   112,   113,    -1,    43,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,    55,    -1,    57,    58,    -1,    -1,    -1,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    -1,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,     3,    55,    -1,    57,    58,    -1,    -1,   112,   113,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    -1,    -1,    77,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,    -1,     3,    55,    -1,    57,    58,    -1,    -1,   112,
     113,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    -1,    -1,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,     3,    55,    -1,    57,    58,    -1,    -1,
     112,   113,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    -1,    -1,    -1,    78,    79,    80,
      -1,    82,    83,    84,    85,    86,    87,    88,    89,    90,
      -1,    -1,    93,    94,    95,    96,    97,    98,    99,   100,
     101,   102,   103,    -1,     3,    55,    -1,    57,    58,    -1,
      -1,   112,   113,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    -1,    -1,    -1,    78,    79,
      80,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    -1,     3,    55,    -1,    57,    58,
      -1,    -1,   112,   113,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    -1,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,    -1,     3,    55,    -1,    57,
      58,    -1,    -1,   112,   113,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    -1,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,     3,    55,    -1,
      57,    58,    -1,    -1,   112,   113,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    -1,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,    -1,     3,    55,
      -1,    57,    58,    -1,    -1,   112,   113,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    -1,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    -1,    -1,
      55,    -1,    57,    58,    -1,    -1,   112,   113,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      -1,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    -1,    -1,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,     4,
       5,     6,     7,     8,    -1,    -1,    -1,   112,   113,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    26,    27,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    43,    44,
      -1,    46,    -1,    48,    -1,    -1,    51,    52,    -1,    -1,
      55,    56,    57,    -1,    59,    -1,    61,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,     4,     5,     6,     7,     8,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    91,    92,    -1,    -1,
      -1,    96,    -1,    26,    27,    -1,    -1,    -1,    -1,   104,
     105,   106,   107,   108,   109,    -1,   111,   112,   113,   114,
      43,    44,    -1,    46,    -1,    48,    -1,    -1,    51,    52,
      -1,    -1,    55,    -1,    57,    -1,    59,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     4,     5,     6,     7,     8,    -1,    81,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,    92,
      -1,    -1,    -1,    96,    26,    27,    -1,   100,    -1,    -1,
      -1,   104,   105,   106,   107,   108,   109,    -1,   111,   112,
     113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,    51,
      52,    -1,    -1,    55,    56,    57,    -1,    59,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,    81,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    91,
      92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,    -1,
      -1,    -1,   104,   105,   106,   107,   108,   109,    -1,   111,
     112,   113,   114,    44,    -1,    46,    -1,    48,    -1,    -1,
      51,    52,    -1,    -1,    55,    56,    57,    -1,    59,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,    -1,
      81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,    -1,
      -1,    -1,    -1,   104,   105,   106,   107,   108,   109,    -1,
     111,   112,   113,   114,    44,    -1,    46,    -1,    48,    -1,
      -1,    51,    52,    -1,    54,    55,    -1,    57,    -1,    59,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,     8,
      -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,    -1,
      -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,   109,
      -1,   111,   112,   113,   114,    44,    -1,    46,    -1,    48,
      -1,    -1,    51,    52,    -1,    -1,    55,    56,    57,    -1,
      59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,     7,
       8,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,    27,
      -1,    -1,    -1,    -1,    -1,   104,   105,   106,   107,   108,
     109,    -1,   111,   112,   113,   114,    44,    -1,    46,    -1,
      48,    -1,    -1,    51,    52,    -1,    54,    55,    -1,    57,
      -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     4,     5,     6,
       7,     8,    -1,    81,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,    26,
      27,    -1,    -1,    -1,    -1,    -1,   104,   105,   106,   107,
     108,   109,    -1,   111,   112,   113,   114,    44,    -1,    46,
      -1,    48,    -1,    -1,    51,    52,    -1,    -1,    55,    -1,
      57,    -1,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    91,    92,    -1,    -1,    -1,    96,
      -1,    -1,    -1,    -1,    -1,    -1,    43,   104,   105,   106,
     107,   108,   109,    -1,   111,   112,   113,   114,    55,    -1,
      57,    58,    -1,    -1,    61,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    -1,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,    43,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   112,   113,    -1,    -1,    55,
      56,    57,    58,    -1,    -1,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    -1,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    43,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   112,   113,    -1,    -1,
      55,    56,    57,    58,    -1,    -1,    -1,    -1,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      -1,    -1,    -1,    78,    79,    80,    -1,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    -1,    -1,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,    -1,
      54,    55,    -1,    57,    58,    -1,    -1,   112,   113,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    -1,    -1,    -1,    78,    79,    80,    -1,    82,    83,
      84,    85,    86,    87,    88,    89,    90,    -1,    -1,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
      -1,    -1,    55,    56,    57,    58,    -1,    -1,   112,   113,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    -1,    -1,    -1,    78,    79,    80,    -1,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,    -1,    -1,    55,    -1,    57,    58,    -1,    -1,   112,
     113,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    -1,    76,    -1,    78,    79,    80,    -1,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    -1,
      -1,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,    -1,    -1,    -1,    -1,    55,    -1,    57,    58,
     112,   113,    61,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    -1,    -1,    -1,    78,
      79,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,    -1,    -1,    55,    56,    57,
      58,    -1,    -1,   112,   113,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    -1,    -1,    -1,
      78,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    -1,    -1,    55,    56,
      57,    58,    -1,    -1,   112,   113,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    -1,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,    -1,    -1,    55,
      -1,    57,    58,    -1,    -1,   112,   113,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    73,    74,    -1,
      -1,    -1,    78,    79,    80,    -1,    82,    83,    84,    85,
      86,    87,    88,    89,    90,    -1,    -1,    93,    94,    95,
      96,    97,    98,    99,   100,   101,   102,   103,    55,    -1,
      57,    58,    -1,    -1,    -1,    -1,   112,   113,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    78,    79,    80,    -1,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    93,    94,    95,    96,
      97,    98,    99,   100,   101,   102,   103,    55,    -1,    57,
      58,    -1,    -1,    -1,    -1,   112,   113,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    79,    80,    -1,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    -1,    -1,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    55,    -1,    57,    58,
      -1,    -1,    -1,    -1,   112,   113,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    80,    -1,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,    55,    -1,    57,    58,    -1,
      -1,    -1,    -1,   112,   113,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    -1,    -1,    93,    94,    95,    96,    97,    98,    99,
     100,   101,   102,   103,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   112,   113
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint16 yystos[] =
{
       0,   116,   117,     0,     1,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    15,    18,    22,    23,
      25,    26,    27,    28,    30,    31,    33,    34,    35,    37,
      38,    39,    40,    41,    42,    44,    46,    47,    48,    49,
      50,    51,    52,    53,    55,    57,    59,    81,    85,    91,
      92,    96,   102,   104,   105,   106,   107,   108,   109,   111,
     112,   113,   114,   118,   119,   121,   122,   123,   125,   126,
     128,   129,   130,   133,   135,   136,   144,   145,   146,   154,
     155,   156,   163,   165,   178,   180,   189,   191,   198,   199,
     200,   202,   204,   212,   213,   214,   216,   217,   219,   222,
     236,   241,   246,   250,   251,   253,   254,   256,   257,   258,
     260,   262,   266,   268,   269,   270,   271,   272,     3,    43,
      62,     3,     1,     6,   124,   253,     1,     6,    44,   256,
       1,     3,     1,     3,    14,     1,   256,     1,   253,   275,
       1,   256,     1,     1,   256,     1,     3,    43,     1,   256,
       1,     6,     1,     6,     1,     3,   256,   247,     1,     6,
       7,     1,   256,   258,     1,     6,     1,     3,     6,   215,
       1,     6,    32,   218,     1,     6,   220,   221,     1,     6,
     261,   267,     1,   256,    56,   256,   273,     1,     3,    43,
       6,   256,    43,    56,    61,   256,   272,   276,   263,   256,
       1,     3,   256,   272,   256,   256,   256,     1,     3,   272,
     256,   256,   256,   256,   256,   256,     4,     6,    26,    58,
     256,   256,     4,   253,   127,    31,    33,    44,   122,   134,
     122,   164,   179,   190,    45,   207,   210,   211,   122,     1,
      55,     3,    55,    57,    58,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    73,    74,    78,    79,    80,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     112,   113,   257,    74,    77,     1,     4,     5,     7,     8,
      51,    52,    96,   120,   252,   256,     3,     3,    77,    74,
       3,    43,     3,    43,     3,     3,     3,     3,    43,     3,
      43,     3,    74,    77,    90,     3,     3,     3,     3,     3,
       3,   122,     3,     3,     3,   223,     3,   242,     3,     3,
       1,     6,   248,   249,     3,     3,     3,     3,     3,     3,
      74,     3,     3,    77,     3,     1,     6,     7,     1,     3,
      32,    77,     3,    74,     3,    77,     3,     1,    55,   264,
     264,     3,     3,     1,    56,    77,   274,     3,   131,   122,
     237,    54,    56,   256,    56,    43,    61,     1,    56,     1,
      56,    77,     1,     6,   205,   206,   265,     3,     3,     3,
       3,     4,     6,    26,   143,   143,   166,   181,   143,     1,
       3,    43,   143,   208,   209,     3,     1,   205,    54,   272,
     100,   256,     6,   256,   256,   256,   256,   256,   256,   256,
     256,   256,   256,   256,   256,     1,   256,   256,   256,   256,
     256,   256,   256,   256,   256,     6,   256,   256,     3,   255,
     255,   255,   255,   255,   255,   255,   255,   255,   255,   255,
     256,   256,     3,     4,     3,   253,   256,   150,   256,   253,
       1,   256,     1,    55,   224,   225,     1,    32,   227,   243,
       3,    77,     1,     1,   252,     6,     3,     3,    75,     3,
      75,     3,     6,     7,     6,     6,     7,   120,   221,     3,
     205,   207,   207,   256,   143,     3,    56,    56,   256,   256,
      56,   256,    61,     1,    61,    77,   207,     9,   122,    16,
      17,   137,   139,   140,   142,     1,     3,    23,    24,   167,
     172,   174,     1,     3,    23,   172,   182,    29,   192,   193,
     194,   195,     3,    43,     9,   143,   122,   203,     1,    54,
       1,    54,   256,    56,    77,     1,    43,   256,   256,   256,
     256,   256,   256,   256,   256,   256,   256,   256,     3,    77,
      74,   148,    76,     3,   147,     3,   205,   231,   227,     3,
       6,   228,   229,     3,   244,     1,   249,     3,     3,     6,
       6,     3,    75,    90,     3,    75,    90,     1,    54,   143,
     143,     9,   238,    43,    56,    61,   206,   143,     3,     1,
       3,     1,   256,     9,   138,   141,     3,     1,     6,     7,
       8,   120,   176,   177,     1,     9,   173,     3,     1,     4,
       6,   187,   188,     9,     1,     3,     4,     6,    90,   196,
     197,     9,   194,   143,     3,     9,    54,   201,     3,    43,
     259,    56,   272,     1,   256,   272,   256,     3,    43,   149,
       1,   256,   149,     1,    54,     3,     6,     9,    36,    44,
      45,   199,   217,   232,   233,   235,     3,    55,   230,    77,
       3,     9,   199,   217,   233,   235,   245,     3,     3,     6,
       6,     6,     6,     3,     9,     9,   132,   256,     3,     6,
       9,   217,   239,   256,   256,    60,     3,     3,     3,     3,
     143,   143,     3,    43,    76,     3,    43,    77,     3,     3,
      43,   175,     3,    43,     3,    43,    77,     3,     3,   253,
       3,    77,    90,     3,     3,    43,    54,    54,     3,    19,
      20,    21,   122,   152,   153,   157,   159,   161,   122,     1,
      77,   151,   226,    74,     3,     3,     6,    54,   272,   229,
       3,     3,     3,     3,     3,     3,     3,    74,    77,   240,
       3,    56,   137,   170,   171,   120,   168,   169,   177,   143,
     122,   185,   186,   183,   184,   188,     3,   197,   253,     3,
       1,     3,    43,     1,     3,    43,     1,     3,    43,     9,
     152,     1,   256,    54,   256,   234,    74,    54,   252,   143,
     122,   143,   122,   143,   122,   143,   122,     3,     3,   158,
     122,     3,   160,   122,     3,   162,   122,     3,     3,   207,
     252,   240,   143,   143,   143,   143,     3,     9,     9,     9,
       9,     3,     3,     3,     3
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
#line 206 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_lone_end ); }
    break;

  case 7:
#line 207 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_case_outside ); }
    break;

  case 8:
#line 211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat)=0; }
    break;

  case 10:
#line 214 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 11:
#line 219 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 12:
#line 224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 13:
#line 229 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addClass( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 14:
#line 234 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (yyvsp[(1) - (1)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(1) - (1)].fal_stat) );
      }
    break;

  case 19:
#line 245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.integer) = - (yyvsp[(2) - (2)].integer); }
    break;

  case 20:
#line 250 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), false );
      }
    break;

  case 21:
#line 256 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getContext() != 0 )
            COMPILER->raiseError(Falcon::e_toplevel_load );
         COMPILER->addLoad( *(yyvsp[(2) - (3)].stringp), true );
      }
    break;

  case 22:
#line 262 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_load );
      }
    break;

  case 23:
#line 268 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->checkLocalUndefined(); (yyval.fal_stat) = (yyvsp[(1) - (1)].fal_stat); }
    break;

  case 24:
#line 269 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_stat)=0;}
    break;

  case 25:
#line 270 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = 0; }
    break;

  case 26:
#line 271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_func ); (yyval.fal_stat) = 0; }
    break;

  case 27:
#line 272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_obj ); (yyval.fal_stat) = 0; }
    break;

  case 28:
#line 273 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_toplevel_class ); (yyval.fal_stat) = 0; }
    break;

  case 29:
#line 274 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syntax ); (yyval.fal_stat) = 0;}
    break;

  case 30:
#line 279 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if( (! COMPILER->isInteractive()) && ((!(yyvsp[(1) - (2)].fal_val)->isExpr()) || (!(yyvsp[(1) - (2)].fal_val)->asExpr()->isStandAlone()) ) )
         {
            Falcon::StmtFunction *func = COMPILER->getFunctionContext();
            if ( func == 0 || ! func->isLambda() )
               COMPILER->raiseError(Falcon::e_noeffect );
         }

         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE, (yyvsp[(1) - (2)].fal_val) );
      }
    break;

  case 31:
#line 290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Value *first = new Falcon::Value( (yyvsp[(1) - (4)].fal_adecl) );
         COMPILER->defineVal( first );
         (yyval.fal_stat) = new Falcon::StmtAutoexpr( LINE,
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, first, (yyvsp[(3) - (4)].fal_val) ) ) );
      }
    break;

  case 32:
#line 296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 50:
#line 330 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_def );
   }
    break;

  case 51:
#line 333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defContext( true );
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr( CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ) ) );
   }
    break;

  case 52:
#line 339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(3) - (5)].fal_val) );
      COMPILER->addStatement( new Falcon::StmtAutoexpr(CURRENT_LINE, new Falcon::Value(
         new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) ) ) );
   }
    break;

  case 53:
#line 348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->defContext( false );  (yyval.fal_stat)=0; }
    break;

  case 54:
#line 350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError( Falcon::e_syn_def ); }
    break;

  case 55:
#line 354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 56:
#line 361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = static_cast<Falcon::StmtWhile *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 57:
#line 368 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtWhile *w = new Falcon::StmtWhile( LINE, (yyvsp[(1) - (2)].fal_val) );
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 58:
#line 376 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 59:
#line 377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while ); (yyval.fal_val) = 0; }
    break;

  case 60:
#line 381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 61:
#line 382 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_while, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 62:
#line 386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         COMPILER->pushLoop( w );
         COMPILER->pushContext( w );
         COMPILER->pushContextSet( &w->children() );
      }
    break;

  case 63:
#line 393 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = static_cast<Falcon::StmtLoop* >(COMPILER->getContext());
         w->setCondition((yyvsp[(6) - (7)].fal_val));
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = w;
      }
    break;

  case 64:
#line 401 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtLoop *w = new Falcon::StmtLoop( LINE );
         if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
            w->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
         (yyval.fal_stat) = w;
      }
    break;

  case 65:
#line 407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_loop );
      (yyval.fal_stat) = 0;
   }
    break;

  case 66:
#line 414 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 67:
#line 415 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(1) - (1)].fal_val); }
    break;

  case 68:
#line 419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->children() );
      }
    break;

  case 69:
#line 427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 70:
#line 434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // use LINE as statement includes EOL
         Falcon::StmtIf *stmt = new Falcon::StmtIf( LINE, (yyvsp[(1) - (2)].fal_val) );
         if( (yyvsp[(2) - (2)].fal_stat) != 0 )
            stmt->children().push_back( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = stmt;
      }
    break;

  case 71:
#line 444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 72:
#line 445 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if ); (yyval.fal_val) = 0; }
    break;

  case 73:
#line 449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 74:
#line 450 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  COMPILER->raiseError(Falcon::e_syn_if, "", CURRENT_LINE ); (yyval.fal_val) = 0; }
    break;

  case 77:
#line 457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         COMPILER->pushContextSet( &stmt->elseChildren() );
      }
    break;

  case 80:
#line 467 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_else ); }
    break;

  case 81:
#line 472 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtIf *stmt = static_cast<Falcon::StmtIf *>(COMPILER->getContext());
         COMPILER->popContextSet();
         Falcon::StmtElif *elif = new Falcon::StmtElif( LINE, (yyvsp[(1) - (1)].fal_val) );
         stmt->elifChildren().push_back( elif );
         COMPILER->pushContextSet( &elif->children() );
      }
    break;

  case 83:
#line 484 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 84:
#line 485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_elif ); (yyval.fal_val) = 0; }
    break;

  case 86:
#line 490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
   }
    break;

  case 87:
#line 497 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_break_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtBreak( LINE );
      }
    break;

  case 88:
#line 506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_break );
         (yyval.fal_stat) = 0;
      }
    break;

  case 89:
#line 514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE );
      }
    break;

  case 90:
#line 524 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->getLoop() == 0 ) {
            COMPILER->raiseError(Falcon::e_continue_out );
            (yyval.fal_stat) = 0;
         }
         else
            (yyval.fal_stat) = new Falcon::StmtContinue( LINE, true );
      }
    break;

  case 91:
#line 533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_continue );
         (yyval.fal_stat) = 0;
      }
    break;

  case 92:
#line 542 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         Falcon::ArrayDecl *decl = (yyvsp[(2) - (4)].fal_adecl);     
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 93:
#line 553 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 94:
#line 562 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f;
         COMPILER->defineVal( (yyvsp[(2) - (4)].fal_val) );
         Falcon::ArrayDecl *decl = new Falcon::ArrayDecl();
         decl->pushBack( (yyvsp[(2) - (4)].fal_val) );
         f = new Falcon::StmtForin( LINE, decl, (yyvsp[(4) - (4)].fal_val) );
         COMPILER->pushLoop( f );
         COMPILER->pushContext( f );
         COMPILER->pushContextSet( &f->children() );
      }
    break;

  case 95:
#line 575 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
         COMPILER->popLoop();
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = f;
      }
    break;

  case 96:
#line 585 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { delete (yyvsp[(2) - (5)].fal_adecl);
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 97:
#line 590 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_forin );
         (yyval.fal_stat) = 0;
      }
    break;

  case 98:
#line 598 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(2) - (2)].fal_stat) != 0 )
            COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 100:
#line 611 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::RangeDecl* rd = new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val),
            new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(3) - (4)].fal_val))), (yyvsp[(4) - (4)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( rd );
      }
    break;

  case 101:
#line 617 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val), 0 ) );
      }
    break;

  case 102:
#line 621 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(1) - (3)].fal_val), 0, 0 ) );
      }
    break;

  case 103:
#line 627 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 104:
#line 628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 105:
#line 629 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 108:
#line 638 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( (yyvsp[(1) - (1)].fal_stat) != 0 )
         {
            Falcon::StmtForin *f = static_cast<Falcon::StmtForin *>(COMPILER->getContext());
            f->children().push_back( (yyvsp[(1) - (1)].fal_stat) );
         }
      }
    break;

  case 112:
#line 652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 113:
#line 665 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_fordot );
         (yyval.fal_stat) = 0;
      }
    break;

  case 114:
#line 673 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 115:
#line 677 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 116:
#line 683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(2) - (3)].fal_adecl)->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 117:
#line 689 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
         adecl->pushBack( new Falcon::Value( COMPILER->addString( "\n" ) ) );
         (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
      }
    break;

  case 118:
#line 696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 119:
#line 701 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_self_print );
         (yyval.fal_stat) = 0;
      }
    break;

  case 120:
#line 710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::ArrayDecl *adecl = new Falcon::ArrayDecl();
      adecl->pushBack( new Falcon::Value( (yyvsp[(1) - (1)].stringp) ) );
      (yyval.fal_stat) = new Falcon::StmtSelfPrint( LINE, adecl );
   }
    break;

  case 121:
#line 719 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 122:
#line 731 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 123:
#line 733 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 124:
#line 742 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forfirst ); }
    break;

  case 125:
#line 746 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 126:
#line 758 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 127:
#line 759 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 128:
#line 768 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_forlast ); }
    break;

  case 129:
#line 772 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 130:
#line 786 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->popContextSet(); }
    break;

  case 131:
#line 788 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 132:
#line 797 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_formiddle ); }
    break;

  case 133:
#line 801 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSwitch *stmt = new Falcon::StmtSwitch( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 134:
#line 809 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 135:
#line 818 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 136:
#line 820 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_switch_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 139:
#line 829 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_switch_body ); }
    break;

  case 141:
#line 835 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 143:
#line 845 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 144:
#line 853 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 145:
#line 857 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 147:
#line 869 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 148:
#line 879 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 150:
#line 888 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 154:
#line 902 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_default_decl ); }
    break;

  case 156:
#line 906 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
      }
    break;

  case 159:
#line 918 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSwitch *stmt = static_cast<Falcon::StmtSwitch *>(COMPILER->getContext());
         // todo: correct error
         if ( stmt->nilBlock() != -1 )
            COMPILER->raiseError(Falcon::e_switch_clash, "nil entry", CURRENT_LINE );
         stmt->nilBlock( stmt->currentBlock() );
      }
    break;

  case 160:
#line 927 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 161:
#line 939 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 162:
#line 950 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 163:
#line 961 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 164:
#line 981 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtSelect *stmt = new Falcon::StmtSelect( LINE, (yyvsp[(1) - (1)].fal_val) );
         COMPILER->pushContext( stmt );
         COMPILER->pushContextSet( &stmt->blocks() );
      }
    break;

  case 165:
#line 989 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContext();
         COMPILER->popContextSet();
         (yyval.fal_stat) = stmt;
      }
    break;

  case 166:
#line 998 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = (yyvsp[(2) - (3)].fal_val); }
    break;

  case 167:
#line 1000 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_select_decl );
         (yyval.fal_val) = 0;
      }
    break;

  case 170:
#line 1009 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_select_body ); }
    break;

  case 172:
#line 1015 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 174:
#line 1025 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 175:
#line 1034 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 176:
#line 1038 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

         Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 178:
#line 1050 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_case_decl );

        Falcon::StmtSelect *stmt = static_cast<Falcon::StmtSelect *>(COMPILER->getContext());
         COMPILER->popContextSet();

         Falcon::StmtCaseBlock *lst = new Falcon::StmtCaseBlock( CURRENT_LINE );
         COMPILER->pushContextSet( &lst->children() );
         stmt->addBlock( lst );
      }
    break;

  case 179:
#line 1060 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->addStatement( (yyvsp[(5) - (5)].fal_stat) );
      }
    break;

  case 183:
#line 1074 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 184:
#line 1086 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 185:
#line 1106 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::StmtTry *t = new Falcon::StmtTry( CURRENT_LINE );
      if ( (yyvsp[(3) - (3)].fal_stat) != 0 )
          t->children().push_back( (yyvsp[(3) - (3)].fal_stat) );
      (yyval.fal_stat) = t;
   }
    break;

  case 186:
#line 1113 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtTry *t = new Falcon::StmtTry( LINE );
         COMPILER->pushContext( t );
         COMPILER->pushContextSet( &t->children() );
      }
    break;

  case 187:
#line 1123 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->popContext();
         COMPILER->popContextSet();
      }
    break;

  case 189:
#line 1132 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_try ); }
    break;

  case 195:
#line 1152 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 196:
#line 1170 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 197:
#line 1190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, 0 );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 198:
#line 1200 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet(); // popping previous catch

         Falcon::StmtTry *t = static_cast<Falcon::StmtTry *>( COMPILER->getContext() );
         COMPILER->defineVal( (yyvsp[(4) - (5)].fal_val) );
         Falcon::StmtCatchBlock *lst = new Falcon::StmtCatchBlock( LINE, (yyvsp[(4) - (5)].fal_val) );
         COMPILER->pushContextSet( &lst->children() );
         t->addHandler( lst );
      }
    break;

  case 199:
#line 1211 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError( Falcon::e_syn_catch );
   }
    break;

  case 202:
#line 1224 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 203:
#line 1236 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 204:
#line 1258 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtRaise( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 205:
#line 1259 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_raise ); (yyval.fal_stat) = 0; }
    break;

  case 206:
#line 1271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 207:
#line 1277 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(2) - (2)].fal_stat) );
         (yyval.fal_stat) = COMPILER->getContext();
         COMPILER->closeFunction();
      }
    break;

  case 209:
#line 1286 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 210:
#line 1287 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 211:
#line 1290 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_funcdecl ); }
    break;

  case 213:
#line 1295 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 214:
#line 1296 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 215:
#line 1303 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 219:
#line 1364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 221:
#line 1381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 222:
#line 1387 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 223:
#line 1392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
         COMPILER->pushContextSet( &func->staticBlock() );
         COMPILER->staticPrefix( &func->symbol()->name() );
      }
    break;

  case 224:
#line 1398 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addStatement( (yyvsp[(3) - (3)].fal_stat) );
         COMPILER->popContextSet();
         COMPILER->staticPrefix(0);
      }
    break;

  case 226:
#line 1407 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static ); }
    break;

  case 228:
#line 1412 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_static, "", CURRENT_LINE ); }
    break;

  case 229:
#line 1422 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = new Falcon::StmtLaunch( LINE, (yyvsp[(2) - (3)].fal_val) );
      }
    break;

  case 230:
#line 1425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_launch ); (yyval.fal_stat) = 0; }
    break;

  case 231:
#line 1434 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 232:
#line 1444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_inv_const_val );
         (yyval.fal_stat) = 0;
      }
    break;

  case 233:
#line 1449 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_const );
         (yyval.fal_stat) = 0;
      }
    break;

  case 234:
#line 1461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         else
            COMPILER->sourceTree()->setExportAll();
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 235:
#line 1470 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         if ( COMPILER->sourceTree()->isExportAll() )
            COMPILER->raiseError(Falcon::e_export_all );
         // no effect
         (yyval.fal_stat) = 0;
      }
    break;

  case 236:
#line 1477 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_export );
         (yyval.fal_stat) = 0;
      }
    break;

  case 237:
#line 1485 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(1) - (1)].stringp) );
         sym->exported(true);
      }
    break;

  case 238:
#line 1490 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Symbol *sym = COMPILER->addGlobalSymbol( (yyvsp[(3) - (3)].stringp) );
         sym->exported(true);
      }
    break;

  case 239:
#line 1498 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (3)].fal_genericList) );
         (yyval.fal_stat) = 0;
      }
    break;

  case 240:
#line 1503 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 241:
#line 1508 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (5)].fal_genericList), (yyvsp[(4) - (5)].stringp), 0, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 242:
#line 1513 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 243:
#line 1533 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 244:
#line 1552 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 245:
#line 1557 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->importSymbols( (yyvsp[(2) - (7)].fal_genericList), (yyvsp[(4) - (7)].stringp), (yyvsp[(6) - (7)].stringp), true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 246:
#line 1562 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 247:
#line 1567 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 248:
#line 1581 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 249:
#line 1586 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (4)].stringp), "", true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 250:
#line 1591 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, false );
         (yyval.fal_stat) = 0;
      }
    break;

  case 251:
#line 1596 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addNamespace( *(yyvsp[(3) - (6)].stringp), *(yyvsp[(5) - (6)].stringp), true, true );
         (yyval.fal_stat) = 0;
      }
    break;

  case 252:
#line 1601 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_import );
         (yyval.fal_stat) = 0;
      }
    break;

  case 253:
#line 1610 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addAttribute( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val), LINE );
     }
    break;

  case 254:
#line 1615 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->raiseError(Falcon::e_syn_attrdecl );
     }
    break;

  case 255:
#line 1622 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::List *lst = new Falcon::List;
         lst->pushBack( new Falcon::String( *(yyvsp[(1) - (1)].stringp) ) );
         (yyval.fal_genericList) = lst;
      }
    break;

  case 256:
#line 1628 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyvsp[(1) - (3)].fal_genericList)->pushBack( new Falcon::String( *(yyvsp[(3) - (3)].stringp) ) );
         (yyval.fal_genericList) = (yyvsp[(1) - (3)].fal_genericList);
      }
    break;

  case 257:
#line 1640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // no effect
         (yyval.fal_stat)=0;
      }
    break;

  case 258:
#line 1645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_directive );
         (yyval.fal_stat)=0;
     }
    break;

  case 261:
#line 1658 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 262:
#line 1662 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), *(yyvsp[(3) - (3)].stringp) );
      }
    break;

  case 263:
#line 1666 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->setDirective( *(yyvsp[(1) - (3)].stringp), (yyvsp[(3) - (3)].integer) );
      }
    break;

  case 264:
#line 1679 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 1711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         // check for expressions in from clauses
         COMPILER->checkLocalUndefined();

         Falcon::StmtClass *cls = static_cast<Falcon::StmtClass *>((yyval.fal_stat));

         // if the class has no constructor, create one in case of inheritance.
         if( cls != 0 )
         {
            if ( cls->ctorFunction() == 0  )
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
      }
    break;

  case 267:
#line 1745 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_class );
      }
    break;

  case 270:
#line 1753 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 271:
#line 1754 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_class, COMPILER->tempLine(), CTX_LINE );
      }
    break;

  case 276:
#line 1771 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 277:
#line 1794 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = 0; }
    break;

  case 278:
#line 1795 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val)=0; }
    break;

  case 279:
#line 1797 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = (yyvsp[(2) - (3)].fal_adecl) == 0 ? 0 : new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
   }
    break;

  case 283:
#line 1810 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 284:
#line 1813 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 287:
#line 1836 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
            COMPILER->pushFunctionContext( func );
            COMPILER->pushContextSet( &func->statements() );
            COMPILER->pushFunction( func->symbol()->getFuncDef() );
         }
      }
    break;

  case 288:
#line 1861 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->popContext();
         COMPILER->popContextSet();
         COMPILER->popFunction();
         COMPILER->popFunctionContext();
      }
    break;

  case 289:
#line 1871 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 290:
#line 1893 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 291:
#line 1926 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 292:
#line 1960 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_stat) = COMPILER->getContext();

         COMPILER->popContext();
         //We didn't pushed a context set
         COMPILER->popFunction();
      }
    break;

  case 296:
#line 1977 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (4)].stringp), (yyvsp[(3) - (4)].fal_val) );
      }
    break;

  case 298:
#line 1982 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->addEnumerator( *(yyvsp[(1) - (2)].stringp) );
      }
    break;

  case 301:
#line 1997 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 302:
#line 2037 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 304:
#line 2065 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_object );
      }
    break;

  case 308:
#line 2077 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->addFunction( (yyvsp[(1) - (1)].fal_stat) );
   }
    break;

  case 309:
#line 2080 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 312:
#line 2109 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::StmtGlobal *glob = new Falcon::StmtGlobal( CURRENT_LINE );
         COMPILER->pushContext( glob );
      }
    break;

  case 313:
#line 2114 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 315:
#line 2128 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 316:
#line 2133 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 318:
#line 2139 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError( Falcon::e_syn_global );
      }
    break;

  case 319:
#line 2146 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         // we create (or retrieve) a globalized symbol
         Falcon::Symbol *sym = COMPILER->globalize( (yyvsp[(1) - (1)].stringp) );

         // then we add the symbol to the global statement (it's just for symbolic asm generation).
         Falcon::StmtGlobal *glob = static_cast<Falcon::StmtGlobal *>( COMPILER->getContext() );
         glob->addSymbol( sym );
      }
    break;

  case 320:
#line 2161 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn(LINE, 0); }
    break;

  case 321:
#line 2162 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_stat) = new Falcon::StmtReturn( LINE, (yyvsp[(2) - (3)].fal_val) ); }
    break;

  case 322:
#line 2163 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->raiseError(Falcon::e_syn_return ); (yyval.fal_stat) = 0; }
    break;

  case 323:
#line 2171 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 324:
#line 2172 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 325:
#line 2173 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 326:
#line 2174 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 327:
#line 2175 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 328:
#line 2176 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 329:
#line 2180 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); }
    break;

  case 330:
#line 2181 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( true ); }
    break;

  case 331:
#line 2182 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( false ); }
    break;

  case 332:
#line 2183 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].integer) ); }
    break;

  case 333:
#line 2184 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].numeric) ); }
    break;

  case 334:
#line 2185 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].stringp) ); }
    break;

  case 335:
#line 2190 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
#line 2208 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setSelf(); }
    break;

  case 338:
#line 2209 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 343:
#line 2237 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( (yyvsp[(2) - (2)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 344:
#line 2238 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, "%d", (int)(yyvsp[(2) - (2)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 345:
#line 2239 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString("self") ); /* do not add the symbol to the compiler */ }
    break;

  case 346:
#line 2240 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyvsp[(3) - (3)].stringp)->prepend( "." ); (yyval.fal_val)->setLBind( (yyvsp[(3) - (3)].stringp) ); /* do not add the symbol to the compiler */ }
    break;

  case 347:
#line 2241 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { char space[32]; sprintf(space, ".%d", (int)(yyvsp[(3) - (3)].integer) ); (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(space) ); }
    break;

  case 348:
#line 2242 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value(); (yyval.fal_val)->setLBind( COMPILER->addString(".self") ); /* do not add the symbol to the compiler */ }
    break;

  case 349:
#line 2243 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neg, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 350:
#line 2244 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_fbind, new Falcon::Value((yyvsp[(1) - (3)].stringp)), (yyvsp[(3) - (3)].fal_val)) ); }
    break;

  case 351:
#line 2245 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            // is this an immediate string sum ?
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isString() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str += *(yyvsp[(4) - (4)].fal_val)->asString();
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.writeNumber( (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_plus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
         }
    break;

  case 352:
#line 2271 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_minus, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 353:
#line 2272 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( (yyvsp[(1) - (4)].fal_val)->asString()->length() );
                  for( int i = 0; i < (yyvsp[(4) - (4)].fal_val)->asInteger(); ++i )
                  {
                     str.append( *(yyvsp[(1) - (4)].fal_val)->asString()  );
                  }
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_times, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 354:
#line 2292 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if( (yyvsp[(1) - (4)].fal_val)->asString()->length() == 0 )
               {
                  COMPILER->raiseError( Falcon::e_invop );
               }
               else {
               
                  if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
                  {
                     Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                     str.setCharAt( str.length()-1, str.getCharAt(str.length()-1) + (yyvsp[(4) - (4)].fal_val)->asInteger() );
                     (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                     delete (yyvsp[(4) - (4)].fal_val);
                     (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
                  }
                  else
                     (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
               }
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_divide, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 355:
#line 2316 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            if ( (yyvsp[(1) - (4)].fal_val)->isString() )
            {
               if ( (yyvsp[(4) - (4)].fal_val)->isInteger() )
               {
                  Falcon::String str( *(yyvsp[(1) - (4)].fal_val)->asString() );
                  str.append( (Falcon::uint32) (yyvsp[(4) - (4)].fal_val)->asInteger() );
                  (yyvsp[(1) - (4)].fal_val)->setString( COMPILER->addString( str ) );
                  delete (yyvsp[(4) - (4)].fal_val);
                  (yyval.fal_val) = (yyvsp[(1) - (4)].fal_val);
               }
               else
                  (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
            }
            else
               (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_modulo, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) );
      }
    break;

  case 356:
#line 2333 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_power, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 357:
#line 2334 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_and, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 358:
#line 2335 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_or, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 359:
#line 2336 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_xor, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 360:
#line 2337 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_left, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 361:
#line 2338 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_shift_right, (yyvsp[(1) - (4)].fal_val), (yyvsp[(4) - (4)].fal_val) ) ); }
    break;

  case 362:
#line 2339 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_bin_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 363:
#line 2340 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_neq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 364:
#line 2341 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_inc, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 365:
#line 2342 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_inc, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 366:
#line 2343 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_post_dec, (yyvsp[(1) - (2)].fal_val) ) ); }
    break;

  case 367:
#line 2344 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_pre_dec, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 368:
#line 2345 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eq, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 369:
#line 2346 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_gt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 370:
#line 2347 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lt, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 371:
#line 2348 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ge, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 372:
#line 2349 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_le, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 373:
#line 2350 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_and, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 374:
#line 2351 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_or, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 375:
#line 2352 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_not, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 376:
#line 2353 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_in, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 377:
#line 2354 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_notin, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 378:
#line 2355 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_provides, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) ) ); }
    break;

  case 379:
#line 2356 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (2)].fal_val) ); }
    break;

  case 380:
#line 2357 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (Falcon::Value *) 0 ); }
    break;

  case 381:
#line 2358 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_strexpand, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 382:
#line 2359 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_indirect, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 383:
#line 2360 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_eval, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 384:
#line 2361 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_oob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 385:
#line 2362 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_deoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 386:
#line 2363 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_isoob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 387:
#line 2364 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_xoroob, (yyvsp[(2) - (2)].fal_val) ) ); }
    break;

  case 394:
#line 2372 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (2)].fal_val), (yyvsp[(2) - (2)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 395:
#line 2377 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( (yyvsp[(1) - (1)].fal_adecl) );
   }
    break;

  case 396:
#line 2381 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_access, (yyvsp[(1) - (4)].fal_val), (yyvsp[(3) - (4)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( exp );
   }
    break;

  case 397:
#line 2386 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_array_byte_access, (yyvsp[(1) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) );
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 398:
#line 2392 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         Falcon::Expression *exp = new Falcon::Expression( Falcon::Expression::t_obj_access, (yyvsp[(1) - (3)].fal_val), new Falcon::Value( (yyvsp[(3) - (3)].stringp) ) );
         if ( (yyvsp[(3) - (3)].stringp)->getCharAt(0) == '_' && ! (yyvsp[(1) - (3)].fal_val)->isSelf() )
         {
            COMPILER->raiseError(Falcon::e_priv_access, COMPILER->tempLine() );
         }
         (yyval.fal_val) = new Falcon::Value( exp );
      }
    break;

  case 401:
#line 2404 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (3)].fal_val) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) );
   }
    break;

  case 402:
#line 2409 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      COMPILER->defineVal( (yyvsp[(1) - (5)].fal_val) );
      (yyvsp[(5) - (5)].fal_adecl)->pushFront( (yyvsp[(3) - (5)].fal_val) );
      Falcon::Value *second = new Falcon::Value( (yyvsp[(5) - (5)].fal_adecl) );
      (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_assign, (yyvsp[(1) - (5)].fal_val), second ) );
   }
    break;

  case 403:
#line 2416 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aadd, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 404:
#line 2417 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_asub, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 405:
#line 2418 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amul, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 406:
#line 2419 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_adiv, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 407:
#line 2420 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_amod, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 408:
#line 2421 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_apow, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 409:
#line 2422 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_aband, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 410:
#line 2423 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 411:
#line 2424 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_abxor, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 412:
#line 2425 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashl, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 413:
#line 2426 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_ashr, (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ) ); }
    break;

  case 414:
#line 2427 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {(yyval.fal_val)=(yyvsp[(2) - (3)].fal_val);}
    break;

  case 415:
#line 2432 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ) ) );
      }
    break;

  case 416:
#line 2435 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (4)].fal_val) ) );
      }
    break;

  case 417:
#line 2438 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( new Falcon::Value( (Falcon::int64) 0 ), (yyvsp[(3) - (4)].fal_val) ) );
      }
    break;

  case 418:
#line 2441 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (5)].fal_val), (yyvsp[(4) - (5)].fal_val) ) );
      }
    break;

  case 419:
#line 2444 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::RangeDecl( (yyvsp[(2) - (7)].fal_val), (yyvsp[(4) - (7)].fal_val), (yyvsp[(6) - (7)].fal_val) ) );
      }
    break;

  case 420:
#line 2451 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall,
                                      (yyvsp[(1) - (4)].fal_val), new Falcon::Value( (yyvsp[(3) - (4)].fal_adecl) ) ) );
      }
    break;

  case 421:
#line 2457 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_funcall, (yyvsp[(1) - (3)].fal_val), 0 ) );
      }
    break;

  case 422:
#line 2461 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { COMPILER->tempLine( CURRENT_LINE ); }
    break;

  case 423:
#line 2462 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(3) - (6)].fal_adecl);
         COMPILER->raiseContextError(Falcon::e_syn_funcall, COMPILER->tempLine(), CTX_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 424:
#line 2471 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 425:
#line 2506 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            (yyval.fal_val) = COMPILER->closeClosure();
         }
    break;

  case 426:
#line 2514 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 427:
#line 2548 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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

  case 429:
#line 2567 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 430:
#line 2571 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 432:
#line 2579 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError(Falcon::e_syn_funcdecl, LINE, CTX_LINE );
      }
    break;

  case 433:
#line 2583 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseError(Falcon::e_syn_funcdecl );
      }
    break;

  case 434:
#line 2590 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
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
         COMPILER->lexer()->pushContext( Falcon::SrcLexer::ct_inner, COMPILER->lexer()->line() );
      }
    break;

  case 435:
#line 2624 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
            COMPILER->lexer()->popContext();
            Falcon::StmtFunction *func = static_cast<Falcon::StmtFunction *>(COMPILER->getContext());
            (yyval.fal_val) = new Falcon::Value( new Falcon::Expression( Falcon::Expression::t_lambda ,
               new Falcon::Value( func->symbol() ) ) );
            // analyze func in previous context.
            COMPILER->closeFunction();
         }
    break;

  case 436:
#line 2640 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      (yyval.fal_val) = new Falcon::Value( new
         Falcon::Expression( Falcon::Expression::t_iif, (yyvsp[(1) - (5)].fal_val), (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ) );
   }
    break;

  case 437:
#line 2645 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (5)].fal_val);
      delete (yyvsp[(3) - (5)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 438:
#line 2652 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
      delete (yyvsp[(1) - (4)].fal_val);
      delete (yyvsp[(3) - (4)].fal_val);
      COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
      (yyval.fal_val) = new Falcon::Value;
   }
    break;

  case 439:
#line 2659 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         delete (yyvsp[(1) - (3)].fal_val);
         COMPILER->raiseError(Falcon::e_syn_iif, CURRENT_LINE );
         (yyval.fal_val) = new Falcon::Value;
      }
    break;

  case 440:
#line 2668 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); }
    break;

  case 441:
#line 2670 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 442:
#line 2674 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_adecl) = (yyvsp[(2) - (3)].fal_adecl);
      }
    break;

  case 443:
#line 2681 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::ArrayDecl() ); }
    break;

  case 444:
#line 2683 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 445:
#line 2687 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_arraydecl, CURRENT_LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_adecl) );
      }
    break;

  case 446:
#line 2695 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {  (yyval.fal_val) = new Falcon::Value( new Falcon::DictDecl() ); }
    break;

  case 447:
#line 2696 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (3)].fal_ddecl) ); }
    break;

  case 448:
#line 2698 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->raiseContextError( Falcon::e_syn_dictdecl, LINE, CTX_LINE );
         (yyval.fal_val) = new Falcon::Value( (yyvsp[(2) - (4)].fal_ddecl) );
      }
    break;

  case 449:
#line 2705 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 450:
#line 2706 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 451:
#line 2710 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_adecl) = new Falcon::ArrayDecl(); (yyval.fal_adecl)->pushBack( (yyvsp[(1) - (1)].fal_val) ); }
    break;

  case 452:
#line 2711 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) ); (yyval.fal_adecl) = (yyvsp[(1) - (3)].fal_adecl); }
    break;

  case 455:
#line 2718 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(1) - (1)].fal_val) );
         Falcon::ArrayDecl *ad = new Falcon::ArrayDecl();
         ad->pushBack( (yyvsp[(1) - (1)].fal_val) );
         (yyval.fal_adecl) = ad;
      }
    break;

  case 456:
#line 2724 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    {
         COMPILER->defineVal( (yyvsp[(3) - (3)].fal_val) );
         (yyvsp[(1) - (3)].fal_adecl)->pushBack( (yyvsp[(3) - (3)].fal_val) );
      }
    break;

  case 457:
#line 2731 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyval.fal_ddecl) = new Falcon::DictDecl(); (yyval.fal_ddecl)->pushBack( (yyvsp[(1) - (3)].fal_val), (yyvsp[(3) - (3)].fal_val) ); }
    break;

  case 458:
#line 2732 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
    { (yyvsp[(1) - (5)].fal_ddecl)->pushBack( (yyvsp[(3) - (5)].fal_val), (yyvsp[(5) - (5)].fal_val) ); (yyval.fal_ddecl) = (yyvsp[(1) - (5)].fal_ddecl); }
    break;


/* Line 1267 of yacc.c.  */
#line 6668 "/home/gian/Progetti/falcon/core/engine/src_parser.cpp"
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


#line 2736 "/home/gian/Progetti/falcon/core/engine/src_parser.yy"
 /* c code */


void flc_src_error (const char *s)  /* Called by yyparse on error */
{
   /* do nothing: manage it in the action */
}

/* end of src_parser.yy */


